"""Shared helpers for per-head attention export."""

from typing import List


def parse_head_indices(spec: str, n_heads: int) -> List[int]:
    """Parse 'all' or comma-separated indices, e.g. '0,3,7'."""
    raw = (spec or "all").strip().lower()
    if raw in {"", "all"}:
        return list(range(int(n_heads)))
    out = [int(x.strip()) for x in spec.split(",") if x.strip()]
    for h in out:
        if h < 0 or h >= n_heads:
            raise ValueError(f"head index {h} out of range [0, {n_heads - 1}]")
    return sorted(set(out))
