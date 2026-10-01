"""Validate, collect, resume, and save private same-state outcome pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .common.answers import last_boxed_content, math_equivalent, require_math_verify
from .common.pairs import PAIR_SCHEMA_VERSION, save_pairs
from .common.runtime import (
    atomic_json,
    atomic_torch_save,
    file_identity,
    load_nemotron,
    native_generate,
    prompt_ids,
    read_jsonl,
)
from .generation.candidates import (
    GumbelNoiseGenerator,
    LogProbScorer,
    build_diverse_transfer_candidates,
)
from .generation.proposal import NativeGenerationSession, ProposalDecision, ProposalStep


# Candidate capture and forced branch rollouts.


@dataclass(frozen=True)
class CapturedCandidateStep:
    """CPU snapshot of one native proposal opportunity and its candidates.

    Attributes:
        block_index: Diffusion block the opportunity belongs to.
        step_index: Decode step within the block.
        nfe: Native function-evaluation counter at capture time.
        noisy_tokens: Noisy block tokens at the native decision point.
        candidates: Candidate token blocks; row ``0`` is the native decision.
        transfer_index: Boolean mask of transfer positions in the native
            decision.
        proposal_scores: Per-candidate proposal scores normalized by the
            transfer count; row ``0`` is the native decision.
        diversity_stats: Counters describing candidate diversity filtering.
        state_hash: SHA-256 hash of the noisy sequence identifying the state.
        hidden_states: Optional frozen proposal hidden states for the state.
    """

    block_index: int
    step_index: int
    nfe: int
    noisy_tokens: torch.Tensor
    candidates: torch.Tensor
    transfer_index: torch.Tensor
    proposal_scores: torch.Tensor
    diversity_stats: dict[str, int]
    state_hash: str
    hidden_states: torch.Tensor | None = None

    @property
    def best_alternative_penalty(self) -> float:
        """Returns the proposal-score gap to the strongest alternative.

        Larger values mean the native choice was clearly better than every
        recorded alternative; smaller values mark more ambiguous branch
        points.

        Returns:
            Gap between the native score and the best recorded alternative;
            ``inf`` when fewer than two candidates were recorded.
        """
        if self.proposal_scores.numel() < 2:
            return float("inf")
        return float(self.proposal_scores[0] - self.proposal_scores[1:].max())


class CandidateTraceHook:
    """Preserve Native decisions while recording real rerank candidates.

    Args:
        num_candidates: Number of candidates recorded per opportunity; the
            native decision plus at least one alternative.
        proposal_temperature: Temperature for alternative-token sampling.
        proposal_noise_scale: Gumbel-noise scale for alternative sampling.
        capture_hidden_states: Whether to also capture frozen proposal
            hidden states for every recorded step.
    """

    def __init__(
        self,
        *,
        num_candidates: int = 4,
        proposal_temperature: float = 0.2,
        proposal_noise_scale: float = 1.0,
        capture_hidden_states: bool = False,
    ) -> None:
        if num_candidates < 2:
            raise ValueError("candidate tracing requires at least two candidates")
        self.num_candidates = num_candidates
        self.capture_hidden_states = capture_hidden_states
        self.generator = GumbelNoiseGenerator(
            temperature=proposal_temperature,
            noise_scale=proposal_noise_scale,
        )
        self.scorer = LogProbScorer()
        self.steps: list[CapturedCandidateStep] = []

    @torch.no_grad()
    def __call__(self, step: ProposalStep) -> None:
        """Records one proposal opportunity and its rerank candidates.

        Args:
            step: Native proposal step to observe.
        """
        hidden_states = step.hidden_states
        if self.capture_hidden_states and hidden_states is None:
            raise RuntimeError(
                "candidate tracing requested hidden states, but the proposal "
                "adapter did not capture them"
            )
        raw_candidates = self.generator.generate_hybrid(step, self.num_candidates)
        candidates, diversity_stats = build_diverse_transfer_candidates(
            step,
            raw_candidates,
            self.num_candidates,
        )
        transfer_count = int(step.native_decision.transfer_index.sum().item())
        proposal_scores = self.scorer.score(step, candidates) / transfer_count
        state = (
            step.sequence_tokens.detach()
            .to(device="cpu", dtype=torch.int32)
            .contiguous()
        )
        self.steps.append(
            CapturedCandidateStep(
                block_index=step.block_index,
                step_index=step.step_index,
                nfe=step.nfe,
                noisy_tokens=step.block_tokens.detach().cpu().squeeze(0).clone(),
                candidates=candidates.detach().cpu().clone(),
                transfer_index=(
                    step.native_decision.transfer_index.detach().cpu().clone()
                ),
                proposal_scores=proposal_scores.detach().float().cpu().clone(),
                diversity_stats=dict(diversity_stats),
                state_hash=hashlib.sha256(state.numpy().tobytes()).hexdigest(),
                hidden_states=(
                    hidden_states.detach().cpu().squeeze(0).clone()
                    if hidden_states is not None
                    else None
                ),
            )
        )

    def ranked_branch_points(self, limit: int) -> list[CapturedCandidateStep]:
        """Return the most ambiguous point per block, then rank globally.

        Args:
            limit: Maximum number of branch points to return.

        Returns:
            The smallest-penalty captured step of every block, sorted by
            ascending ``best_alternative_penalty`` and truncated to
            ``limit`` entries.

        Raises:
            ValueError: If ``limit`` is not positive.
        """

        if limit <= 0:
            raise ValueError("branch-point limit must be positive")
        best_by_block: dict[int, CapturedCandidateStep] = {}
        for step in self.steps:
            current = best_by_block.get(step.block_index)
            if (
                current is None
                or step.best_alternative_penalty < current.best_alternative_penalty
            ):
                best_by_block[step.block_index] = step
        return sorted(
            best_by_block.values(),
            key=lambda step: (
                step.best_alternative_penalty,
                step.block_index,
                step.step_index,
            ),
        )[:limit]


class ForcedCandidateHook:
    """Replay Native until one captured point, then force one candidate.

    Args:
        captured_step: Previously captured step to replay until.
        candidate_index: Non-native candidate row to force, ``1`` or higher.
    """

    def __init__(
        self,
        captured_step: CapturedCandidateStep,
        candidate_index: int,
    ) -> None:
        if not 1 <= candidate_index < captured_step.candidates.size(0):
            raise ValueError("candidate_index must select a non-native candidate")
        self.captured_step = captured_step
        self.candidate_index = candidate_index
        self.forced = False

    @torch.no_grad()
    def __call__(self, step: ProposalStep) -> ProposalDecision | None:
        """Returns the forced decision at the captured point, else ``None``.

        Args:
            step: Native proposal step from the replay.

        Returns:
            Decision forcing the recorded candidate at the captured point,
            or ``None`` while the replay has not reached it yet.

        Raises:
            RuntimeError: If the captured point is reached twice, or the
                native replay diverges before reaching it.
        """
        target = self.captured_step
        if (step.block_index, step.step_index, step.nfe) != (
            target.block_index,
            target.step_index,
            target.nfe,
        ):
            return None
        if self.forced:
            raise RuntimeError("captured candidate point was reached more than once")
        current_transfer = step.native_decision.transfer_index.detach().cpu()
        if not torch.equal(current_transfer, target.transfer_index):
            raise RuntimeError(
                "native replay diverged before the captured candidate point"
            )
        self.forced = True
        candidate = target.candidates[self.candidate_index].to(
            device=step.block_tokens.device,
            dtype=step.block_tokens.dtype,
        )
        return ProposalDecision(
            tokens=candidate.unsqueeze(0),
            transfer_index=step.native_decision.transfer_index.clone(),
        )


# Pair collection CLI.


COLLECTOR_SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    """Parses pair-collection command-line arguments.

    Returns:
        argparse.Namespace: Parsed options; see ``--help`` and the README for
        the reference invocation of every flag.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Pair collection intentionally excludes benchmark test splits.",
    )
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--wrong-branch-points", type=int, default=6)
    parser.add_argument("--correct-branch-points", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--proposal-temperature", type=float, default=0.2)
    parser.add_argument("--proposal-noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260842)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _problem_id(index: int, row: dict[str, Any]) -> str:
    """Builds the stable identifier used for one dataset row.

    Args:
        index: Row position in the dataset.

        row: Dataset row holding an optional ``problem_id``.

    Returns:
        The explicit ``problem_id`` when present, otherwise a
        ``private:<index>:<digest>`` identifier derived from the problem
        text.
    """
    if row.get("problem_id"):
        return str(row["problem_id"])
    digest = hashlib.sha256(str(row["problem"]).encode()).hexdigest()[:16]
    return f"private:{index}:{digest}"


def _rendered_config(args: argparse.Namespace) -> dict[str, Any]:
    """Renders the resume configuration for one collector run.

    Args:
        args: Parsed CLI options.

    Returns:
        Dict capturing every setting that must stay stable across resumes.
    """
    return {
        "schema_version": COLLECTOR_SCHEMA_VERSION,
        "collector": "same_state_candidate_rollout",
        "model": args.model,
        "dataset": file_identity(args.dataset_jsonl),
        "split": args.split,
        "indices": args.indices,
        "limit": args.limit,
        "num_candidates": args.num_candidates,
        "wrong_branch_points": args.wrong_branch_points,
        "correct_branch_points": args.correct_branch_points,
        "max_new_tokens": args.max_new_tokens,
        "block_length": args.block_length,
        "threshold": args.threshold,
        "proposal_temperature": args.proposal_temperature,
        "proposal_noise_scale": args.proposal_noise_scale,
        "seed": args.seed,
    }


def _load_state(args: argparse.Namespace) -> dict[str, Any]:
    """Loads collector state for resume, or initializes a fresh state.

    Args:
        args: Parsed CLI options supplying the output directory.

    Returns:
        Collector state with schema version, completed indices, items,
        pairs, and accumulated walltime.

    Raises:
        FileExistsError: If output exists and matching ``--resume`` is off.

        ValueError: If the persisted configuration or schema does not match
            this run.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = _rendered_config(args)
    config_path = args.output_dir / "run_config.json"
    state_path = args.output_dir / "collector_state.pt"
    if config_path.exists() or state_path.exists():
        if not args.resume or not config_path.exists() or not state_path.exists():
            raise FileExistsError(
                "existing collector output requires matching --resume"
            )
        if json.loads(config_path.read_text()) != config:
            raise ValueError("resume configuration does not match existing output")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("schema_version") != COLLECTOR_SCHEMA_VERSION:
            raise ValueError("unsupported collector state schema")
        return state
    atomic_json(config_path, config)
    return {
        "schema_version": COLLECTOR_SCHEMA_VERSION,
        "completed_indices": [],
        "items": [],
        "pairs": [],
        "walltime_seconds": 0.0,
    }


def _decode(
    tokenizer: Any, output: torch.Tensor, prompt_length: int, gold: str
) -> dict[str, Any]:
    """Decodes one native rollout and judges it against the gold answer.

    Args:
        tokenizer: Tokenizer used to decode generated ids.

        output: Generated token ids including the prompt.

        prompt_length: Number of leading prompt tokens to skip.

        gold: Gold answer used for reward scoring.

    Returns:
        Dict with the boxed ``prediction``, binary ``reward``, and full
        ``final_text``.
    """
    text = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
    return {
        "prediction": last_boxed_content(text),
        "reward": int(math_equivalent(text, gold)),
        "final_text": text,
    }


def _materialize_pairs(
    problem_id: str,
    split: str,
    captured: Any,
    candidates: list[dict[str, Any]],
    candidate_features: torch.Tensor,
    noisy_features: torch.Tensor,
) -> list[dict[str, Any]]:
    """Builds positive/negative pairs from one captured candidate step.

    Args:
        problem_id: Identifier recorded on every pair.

        split: Dataset split recorded on every pair.

        captured: Captured candidate step owning the shared state.

        candidates: Rollout candidate rows with ``reward`` and
            ``proposal_logprob`` fields.

        candidate_features: Token features recorded per candidate.

        noisy_features: Token features of the shared noisy block.

    Returns:
        Pair records crossing every correct candidate with every wrong
        candidate, hardest negative first.
    """
    positives = [row for row in candidates if row["reward"] == 1]
    negatives = sorted(
        (row for row in candidates if row["reward"] == 0),
        key=lambda row: row["proposal_logprob"],
        reverse=True,
    )
    result = []
    for positive in positives:
        for rank, negative in enumerate(negatives):
            result.append(
                {
                    "schema_version": PAIR_SCHEMA_VERSION,
                    "problem_id": problem_id,
                    "split": split,
                    "block_index": captured.block_index,
                    "step_index": captured.step_index,
                    "state_hash": captured.state_hash,
                    "noisy_tokens": captured.noisy_tokens.clone(),
                    "hidden_states": captured.hidden_states.clone(),
                    "noisy_features": noisy_features.clone(),
                    "positive_tokens": captured.candidates[
                        positive["candidate_index"]
                    ].clone(),
                    "negative_tokens": captured.candidates[
                        negative["candidate_index"]
                    ].clone(),
                    "positive_candidate_features": candidate_features[
                        positive["candidate_index"]
                    ].clone(),
                    "negative_candidate_features": candidate_features[
                        negative["candidate_index"]
                    ].clone(),
                    "transfer_mask": captured.transfer_index.squeeze(0).clone(),
                    "positive_logprob": positive["proposal_logprob"],
                    "negative_logprob": negative["proposal_logprob"],
                    "negative_kind": "rollout_hard" if rank == 0 else "rollout_easy",
                }
            )
    return result


def _save_state(args: argparse.Namespace, state: dict[str, Any]) -> None:
    """Persists collector state, pair artifacts, and a run summary.

    Args:
        args: Parsed CLI options supplying the output directory.

        state: Collector state to persist.
    """
    atomic_torch_save(args.output_dir / "collector_state.pt", state)
    if state["pairs"]:
        save_pairs(args.output_dir / "pairs.pt", state["pairs"])
    summary = {
        "completed_problems": len(state["completed_indices"]),
        "materialized_pairs": len(state["pairs"]),
        "native_accuracy": 100.0
        * sum(item["native_reward"] for item in state["items"])
        / len(state["items"]),
        "walltime_seconds": state["walltime_seconds"],
    }
    atomic_json(args.output_dir / "summary.json", summary)


def main() -> None:
    """Runs same-state pair collection with resumable per-problem progress.

    Replays the frozen proposal, captures the most ambiguous branch points,
    forces alternative candidates there, labels outcomes with the math
    scorer, and persists validated pairs plus collector state.
    """
    args = parse_args()
    require_math_verify()
    if args.num_candidates < 2:
        raise ValueError("num-candidates must be at least two")
    if args.block_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens and block-length must be positive")
    if args.max_new_tokens % args.block_length:
        raise ValueError("max-new-tokens must be divisible by block-length")
    rows = read_jsonl(args.dataset_jsonl)
    selected = args.indices or [
        index
        for index, row in enumerate(rows)
        if row.get("split", args.split) == args.split
    ]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("no dataset rows selected")
    if any(index < 0 or index >= len(rows) for index in selected):
        raise IndexError("selected dataset index is out of range")
    if any(rows[index].get("split", args.split) != args.split for index in selected):
        raise ValueError("selected rows must all belong to --split")

    state = _load_state(args)
    completed = set(state["completed_indices"])
    tokenizer, model = load_nemotron(args.model, args.device)
    embedding = model.get_input_embeddings()
    if embedding is None:
        raise RuntimeError("Nemotron does not expose token embeddings")
    eos_id = int(model.config.eos_token_id)

    for index in selected:
        if index in completed:
            continue
        started = time.perf_counter()
        row = rows[index]
        problem_id = _problem_id(index, row)
        inputs = prompt_ids(tokenizer, str(row["problem"]), args.device)
        trace = CandidateTraceHook(
            num_candidates=args.num_candidates,
            proposal_temperature=args.proposal_temperature,
            proposal_noise_scale=args.proposal_noise_scale,
            capture_hidden_states=True,
        )
        item_seed = args.seed + index
        torch.manual_seed(item_seed)
        with NativeGenerationSession(
            model, proposal_hook=trace, capture_hidden_states=True
        ) as generation:
            native_ids, native_nfe = native_generate(
                generation,
                inputs,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                threshold=args.threshold,
                eos_token_id=eos_id,
            )
        native = _decode(tokenizer, native_ids, inputs.size(1), str(row["answer"]))
        limit = (
            args.correct_branch_points if native["reward"] else args.wrong_branch_points
        )
        new_pairs: list[dict[str, Any]] = []
        mixed_states = 0
        branch_rollouts = 0
        for captured in trace.ranked_branch_points(limit):
            candidates = [
                {
                    "candidate_index": 0,
                    "proposal_logprob": float(captured.proposal_scores[0]),
                    "reward": native["reward"],
                }
            ]
            for candidate_index in range(1, captured.candidates.size(0)):
                force = ForcedCandidateHook(captured, candidate_index)
                torch.manual_seed(item_seed)
                with NativeGenerationSession(model, proposal_hook=force) as generation:
                    branch_ids, _ = native_generate(
                        generation,
                        inputs,
                        max_new_tokens=args.max_new_tokens,
                        block_length=args.block_length,
                        threshold=args.threshold,
                        eos_token_id=eos_id,
                    )
                if not force.forced:
                    raise RuntimeError("forced candidate point was not reached")
                branch = _decode(
                    tokenizer, branch_ids, inputs.size(1), str(row["answer"])
                )
                candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "proposal_logprob": float(
                            captured.proposal_scores[candidate_index]
                        ),
                        "reward": branch["reward"],
                    }
                )
                branch_rollouts += 1
            if {row["reward"] for row in candidates} == {0, 1}:
                mixed_states += 1
                with torch.no_grad():
                    candidate_features = embedding(
                        captured.candidates.to(args.device)
                    ).cpu()
                    noisy_features = embedding(
                        captured.noisy_tokens.to(args.device)
                    ).cpu()
                new_pairs.extend(
                    _materialize_pairs(
                        problem_id,
                        args.split,
                        captured,
                        candidates,
                        candidate_features,
                        noisy_features,
                    )
                )
        state["items"].append(
            {
                "dataset_index": index,
                "problem_id": problem_id,
                "native_reward": native["reward"],
                "native_prediction": native["prediction"],
                "native_nfe": native_nfe,
                "candidate_states": len(trace.steps),
                "mixed_candidate_states": mixed_states,
                "branch_rollouts": branch_rollouts,
            }
        )
        state["pairs"].extend(new_pairs)
        state["completed_indices"].append(index)
        state["walltime_seconds"] += time.perf_counter() - started
        _save_state(args, state)
        print(json.dumps(state["items"][-1]), flush=True)


if __name__ == "__main__":
    main()
