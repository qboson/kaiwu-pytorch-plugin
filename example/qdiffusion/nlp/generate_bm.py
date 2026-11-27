"""Generate text with a trained BM-guided QDiffusion checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import sys

import torch


def _bootstrap_repo() -> None:
    """Expose the repository-local ``kaiwu`` namespace to this CLI."""

    repo_root = Path(__file__).resolve().parents[3]
    for path in (str(repo_root / "src"), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import kaiwu

    local_namespace = str(repo_root / "src" / "kaiwu")
    if local_namespace not in kaiwu.__path__:
        kaiwu.__path__.insert(0, local_namespace)


_bootstrap_repo()

from .checkpoint import (  # noqa: E402
    load_bm_weights,
    read_bm_checkpoint,
)
from .models import BMTextGenerator, MDLMBackbone  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse BM generation arguments.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposal-checkpoint",
        default="kuleshov-group/mdlm-owt",
    )
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--bm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=2)
    parser.add_argument("--importance-start-t", type=float, default=1.0)
    parser.add_argument("--importance-end-t", type=float, default=0.0)
    parser.add_argument("--energy-temperature", type=float, default=1.0)
    parser.add_argument("--remask-ratio", type=float, default=0.1)
    parser.add_argument(
        "--weights",
        choices=("ema", "raw"),
        default="ema",
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def build_generator(
    args: argparse.Namespace,
    proposal: MDLMBackbone,
    device: torch.device,
) -> BMTextGenerator:
    """Build a BM generator from checkpoint architecture metadata.

    Args:
        args: Parsed generation arguments.
        proposal: Frozen proposal model used for reverse diffusion.
        device: Device on which proposal and energy modules are constructed.

    Returns:
        Generator with raw or EMA BM weights restored.

    Raises:
        ValueError: If the checkpoint is not BM-based, uses non-identity
            visibles, or was trained with a different proposal checkpoint.
    """

    checkpoint = read_bm_checkpoint(args.bm_checkpoint)
    metadata = checkpoint["metadata"]
    if metadata.get("energy_type", "bm") != "bm":
        raise ValueError("--bm-checkpoint must contain BM energy weights.")
    visible_transform = metadata.get("visible_transform", "identity")
    if visible_transform != "identity":
        raise ValueError(
            "The BM-only example requires continuous identity visibles; "
            f"checkpoint uses {visible_transform!r}."
        )
    trained_proposal = metadata.get(
        "proposal_checkpoint",
        metadata.get("mdlm_checkpoint"),
    )
    if (
        trained_proposal is not None
        and trained_proposal != args.proposal_checkpoint
    ):
        raise ValueError(
            "BM and proposal checkpoints differ: "
            f"{trained_proposal!r} != {args.proposal_checkpoint!r}."
        )
    # The proposal and energy encoder start from the same pretrained model but
    # have different roles. The proposal stays frozen; checkpoint weights are
    # loaded into this separate energy-side copy below.
    energy_backbone = (
        MDLMBackbone.from_pretrained(
            args.proposal_checkpoint,
            tokenizer_name_or_path=args.tokenizer,
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    generator = BMTextGenerator(
        proposal,
        energy_backbone,
        bm_num_visible=int(metadata.get("bm_num_visible", 768)),
        bm_num_hidden=int(metadata.get("bm_num_hidden", 256)),
        bm_sampler_kwargs=metadata.get("sampler_kwargs", {}),
        pooling_mode=metadata.get("pooling_mode", "attention"),
        num_candidates=args.num_candidates,
        energy_temperature=args.energy_temperature,
        device=device,
    )
    load_bm_weights(
        generator,
        checkpoint,
        use_ema=args.weights == "ema",
    )
    generator.energy_model.eval()
    return generator


def main() -> None:
    """Run BM-guided generation and write one JSON record per sample."""

    args = parse_args()
    if args.seed is None:
        args.seed = secrets.randbelow(2**31)
    if args.sequence_length <= 0 or args.steps <= 0:
        raise ValueError("--sequence-length and --steps must be positive.")
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("--num-samples and --batch-size must be positive.")
    if args.num_candidates < 2:
        raise ValueError("--num-candidates must be at least 2.")
    if not torch.cuda.is_available():
        raise RuntimeError("BM generation requires Linux/CUDA.")

    # Seed model construction as well as generation for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    proposal = (
        MDLMBackbone.from_pretrained(
            args.proposal_checkpoint,
            tokenizer_name_or_path=args.tokenizer,
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    generator = build_generator(args, proposal, device)
    # Model construction consumes random numbers. Resetting here makes the
    # candidate stream depend only on the recorded generation seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    proposal_forwards = 0
    guided_steps = 0
    with args.output.open("w", encoding="utf-8") as output_file:
        for batch_start in range(0, args.num_samples, args.batch_size):
            batch_size = min(
                args.batch_size,
                args.num_samples - batch_start,
            )
            # Unconditional generation starts from the absorbing all-mask
            # state expected by the released text-diffusion checkpoint.
            masked = torch.full(
                (batch_size, args.sequence_length),
                proposal.mask_id,
                dtype=torch.long,
                device=device,
            )
            samples = generator.generate(
                masked,
                max_steps=args.steps,
                importance_start_t=args.importance_start_t,
                importance_end_t=args.importance_end_t,
                remask_ratio=args.remask_ratio,
            )
            proposal_forwards += int(
                generator.last_sampling_stats["proposal_forwards"]
            )
            guided_steps += int(
                generator.last_sampling_stats["guided_steps"]
            )
            for token_ids in samples.cpu().tolist():
                output_file.write(
                    json.dumps(
                        {
                            "text": proposal.tokenizer.decode(
                                token_ids,
                                skip_special_tokens=False,
                            ),
                            "token_ids": token_ids,
                            "seed": args.seed,
                            "steps": args.steps,
                            "num_candidates": args.num_candidates,
                            "remask_ratio": args.remask_ratio,
                            "weights": args.weights,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                generated += 1
            output_file.flush()
            print(
                json.dumps(
                    {
                        "generated": generated,
                        "num_samples": args.num_samples,
                    }
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "resolved_seed": args.seed,
                "num_samples": generated,
                "proposal_forwards": proposal_forwards,
                "guided_steps": guided_steps,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
