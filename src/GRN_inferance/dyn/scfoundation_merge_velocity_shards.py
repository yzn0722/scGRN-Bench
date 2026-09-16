#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge scFoundation velocity shard npz files into one velocity_field.npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge scFoundation velocity shard npz files")
    p.add_argument("--shard-glob", required=True, type=str, help='e.g. ".../velocity_field.shard*.npz"')
    p.add_argument("--out-npz", required=True, type=str)
    p.add_argument("--expr-csv", default="", type=str, help="Optional: validate cell order against export CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pattern = Path(args.shard_glob)
    if pattern.parent == Path("."):
        shard_paths = sorted(Path.cwd().glob(pattern.name))
    else:
        shard_paths = sorted(pattern.parent.glob(pattern.name))

    if not shard_paths:
        raise FileNotFoundError(f"No shards matched: {args.shard_glob}")

    parts: list[np.ndarray] = []
    cells_all: list[str] = []
    genes_ref = None

    for sp in shard_paths:
        d = np.load(sp)
        vel = d["vel_cell"].astype(np.float32)
        cells = d["cells"].astype(str).tolist()
        genes = d["genes"].astype(str).tolist()
        if genes_ref is None:
            genes_ref = genes
        elif list(genes_ref) != list(genes):
            raise ValueError(f"Gene mismatch in shard {sp}")
        parts.append(vel)
        cells_all.extend(cells)
        print(f"  shard {sp.name}: {vel.shape[0]} cells", flush=True)

    vel_all = np.vstack(parts)
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, vel_cell=vel_all, cells=np.array(cells_all), genes=np.array(genes_ref))
    print(f"[OK] merged {len(shard_paths)} shards -> {out}  shape={vel_all.shape}", flush=True)

    if args.expr_csv:
        import pandas as pd

        expected = pd.read_csv(args.expr_csv, index_col=0).columns.astype(str).tolist()
        if expected != cells_all:
            print("[WARN] merged cell order differs from expr CSV column order", flush=True)


if __name__ == "__main__":
    main()
