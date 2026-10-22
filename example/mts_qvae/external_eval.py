from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ExternalSchema:
    """Schema for parsing external predictor outputs.

    We intentionally do not hardcode DeepLoc/TargetP formats to avoid inventing
    CLI/output fields. Users can provide a schema JSON.
    """

    id_col: Optional[str] = None
    category_col: Optional[str] = None
    score_col: Optional[str] = None
    extra_cols: Tuple[str, ...] = ()
    inferred: bool = False


def load_schema(path: Path) -> ExternalSchema:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExternalSchema(
        id_col=data.get("id_col"),
        category_col=data.get("category_col"),
        score_col=data.get("score_col"),
        extra_cols=tuple(data.get("extra_cols") or ()),
        inferred=bool(data.get("inferred", False)),
    )


def detect_format(path: Path) -> str:
    """Return one of: 'json', 'table'."""
    head = path.read_text(encoding="utf-8", errors="ignore")[:2048].lstrip()
    if head.startswith("{") or head.startswith("["):
        return "json"
    return "table"


def load_output(path: Path):
    fmt = detect_format(path)
    if fmt == "json":
        return json.loads(path.read_text(encoding="utf-8"))

    # Table: use pandas if available for robust delimiter inference.
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Parsing external tool table output requires pandas; install it or output JSON."
        ) from e

    # Try delimiter inference first; fall back to TSV.
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")
    return df


def infer_schema_from_table(df) -> ExternalSchema:
    cols = [str(c) for c in df.columns]

    def find_first(predicates) -> Optional[str]:
        for c in cols:
            c_low = c.lower()
            if any(p(c_low) for p in predicates):
                return c
        return None

    id_col = find_first(
        [
            lambda s: s in {"id", "seqid", "sequence_id", "name"},
            lambda s: "identifier" in s,
        ]
    )

    category_col = find_first(
        [
            lambda s: s in {"prediction", "pred", "class", "localization", "location"},
            lambda s: "pred" in s and "prob" not in s,
        ]
    )

    score_col = find_first(
        [
            lambda s: s in {"score", "probability", "prob", "confidence"},
            lambda s: s.endswith("_prob"),
        ]
    )

    return ExternalSchema(
        id_col=id_col,
        category_col=category_col,
        score_col=score_col,
        extra_cols=tuple(),
        inferred=True,
    )


def summarize_table(df, schema: ExternalSchema) -> Dict[str, Any]:
    cols = [str(c) for c in df.columns]
    if schema.category_col is None:
        raise ValueError(
            "Cannot summarize external output without category_col. "
            "Provide --external-schema with category_col. Available columns: "
            + ", ".join(cols)
        )

    cat = schema.category_col
    if cat not in df.columns:
        raise ValueError(
            f"category_col={cat!r} not present in output. Available columns: {cols}"
        )

    # Basic distribution over predicted categories
    vc = df[cat].astype(str).value_counts(dropna=False)
    total = int(vc.sum())
    dist = {str(k): float(v) for k, v in vc.to_dict().items()}
    frac = {str(k): (float(v) / max(1, total)) for k, v in vc.to_dict().items()}

    summary: Dict[str, Any] = {
        "n": float(total),
        "category_col": cat,
        "category_counts": dist,
        "category_fractions": frac,
        "columns": cols,
        "schema_inferred": bool(schema.inferred),
    }

    if schema.score_col and schema.score_col in df.columns:
        s = df[schema.score_col]
        try:
            s = s.astype(float)
            summary["score_col"] = schema.score_col
            summary["score_mean"] = float(s.mean())
            summary["score_min"] = float(s.min())
            summary["score_max"] = float(s.max())
        except Exception:
            # keep silent: score column exists but isn't numeric
            summary["score_col"] = schema.score_col

    return summary
