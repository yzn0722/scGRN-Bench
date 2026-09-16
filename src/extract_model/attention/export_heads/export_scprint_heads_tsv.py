import argparse
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from utils_heads import parse_head_indices
from utils_scprint import (
    format_slice_table,
    load_arch_from_ckpt,
    resolve_scprint_head_indices,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run scPRINT with head_agg=none and export per-slice TSV from "
            "AnnData.varp['GRN'] (shape genes x genes x n_slices)."
        )
    )
    _here = Path(__file__).resolve().parent
    p.add_argument(
        "--script-path",
        default=str(_here.parent / "scPRINT.py"),
        type=str,
        help="Path to attention/scPRINT.py",
    )
    _weights = _here.parent.parent.parent.parent / "models" / "weights"
    p.add_argument(
        "--ckpt",
        default=str(_weights / "1lnm8pgh_geneslist_9606.ckpt"),
        type=str,
    )
    p.add_argument(
        "--token-pkl",
        default="/mnt/10T/yzn/scPRINT/data/main/token_dictionary.pkl",
        type=str,
    )
    p.add_argument("--expr-csv", required=True, type=str)
    p.add_argument("--out-prefix", required=True, type=str)
    p.add_argument("--species", default="NCBITaxon:9606", type=str)
    p.add_argument("--cell-type-name", default="all", type=str)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--preprocess", default="softmax", choices=["softmax", "sinkhorn", "none"])
    p.add_argument("--forward-mode", default="none", type=str)
    p.add_argument("--ckpt-gene-emb-offset", default=0, type=int)
    p.add_argument("--topk", default=10, type=int)
    p.add_argument(
        "--grn-key",
        default="GRN",
        type=str,
        help="Key in adata.varp holding [G,G,K] attention tensors (scPRINT default: GRN).",
    )
    p.add_argument(
        "--gene-col",
        default="symbol",
        type=str,
        help="Column in adata.var for gene symbols in output TSV.",
    )
    p.add_argument(
        "--head-indices",
        default="all",
        type=str,
        help="Slice indices on varp GRN last axis, e.g. '0,2,31'. Default 'all'.",
    )
    p.add_argument(
        "--last-layer-only",
        action="store_true",
        help="Export only the last transformer layer (4 heads -> slices 28-31 for default 8L/4H ckpt).",
    )
    p.add_argument(
        "--nlayers",
        type=int,
        default=0,
        help="Transformer layers (0 = read from --ckpt hyper_parameters).",
    )
    p.add_argument(
        "--nhead",
        type=int,
        default=0,
        help="Heads per layer (0 = read from --ckpt).",
    )
    p.add_argument(
        "--print-slice-table",
        action="store_true",
        help="Print layer/head -> slice_index table and exit (no inference).",
    )
    return p.parse_args()


def _gene_names(adata: ad.AnnData, gene_col: str) -> list:
    if gene_col in adata.var.columns:
        return adata.var[gene_col].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _matrix_to_edges(mat: np.ndarray, genes: list) -> pd.DataFrame:
    edges = pd.DataFrame(mat, index=genes, columns=genes).stack().reset_index()
    edges.columns = ["Gene1", "Gene2", "EdgeWeight"]
    edges = edges[edges["Gene1"] != edges["Gene2"]]
    return edges.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)


def export_from_h5ad(
    h5ad_path: Path,
    out_prefix: Path,
    head_indices: str = "all",
    grn_key: str = "GRN",
    gene_col: str = "symbol",
    nlayers: int = 8,
    nhead: int = 4,
    last_layer_only: bool = False,
) -> None:
    adata = ad.read_h5ad(str(h5ad_path))
    genes = _gene_names(adata, gene_col)
    exported = 0

    # scPRINT head_agg=none stores [G, G, L*H] in varp (not layers/uns)
    if grn_key in adata.varp:
        grn = np.asarray(adata.varp[grn_key])
        if grn.ndim != 3:
            raise RuntimeError(f"varp['{grn_key}'] expected 3D, got shape {grn.shape}")
        n_slices = grn.shape[2]
        if len(genes) != grn.shape[0]:
            print(
                f"[WARN] n_genes={len(genes)} != GRN.shape[0]={grn.shape[0]}; "
                "using var index length"
            )
            genes = genes[: grn.shape[0]]
        selected = resolve_scprint_head_indices(
            head_indices, nlayers, nhead, last_layer_only=last_layer_only
        )
        if n_slices != nlayers * nhead:
            print(
                f"[WARN] varp K={n_slices} != nlayers*nhead={nlayers * nhead}; "
                "using indices within [0, K-1]"
            )
            selected = [i for i in selected if 0 <= i < n_slices]
        print(
            f"[INFO] varp['{grn_key}'] shape={grn.shape} "
            f"(nlayers={nlayers}, nhead={nhead}); exporting slices {selected}"
        )
        for h in selected:
            edges = _matrix_to_edges(grn[:, :, h], genes)
            out = Path(f"{out_prefix}_head{h}.tsv")
            edges.to_csv(out, sep="\t", index=False)
            print(f"[INFO] Saved {out} edges={len(edges)}")
            exported += 1

    # Legacy fallbacks
    head_keys = sorted(k for k in adata.layers.keys() if k.lower().startswith("head"))
    if exported == 0 and head_keys:
        n_heads = len(head_keys)
        head_keys = [head_keys[h] for h in parse_head_indices(head_indices, n_heads)]
        for key in head_keys:
            mat = np.asarray(adata.layers[key])
            edges = _matrix_to_edges(mat, genes)
            out = Path(f"{out_prefix}_{key}.tsv")
            edges.to_csv(out, sep="\t", index=False)
            print(f"[INFO] Saved {out} edges={len(edges)}")
            exported += 1

    if exported == 0 and "heads" in adata.uns:
        heads = np.asarray(adata.uns["heads"])
        if heads.ndim == 3:
            for h in parse_head_indices(head_indices, heads.shape[0]):
                edges = _matrix_to_edges(heads[h], genes)
                out = Path(f"{out_prefix}_head{h}.tsv")
                edges.to_csv(out, sep="\t", index=False)
                print(f"[INFO] Saved {out} edges={len(edges)}")
                exported += 1

    if exported == 0:
        raise RuntimeError(
            "Could not find per-head/slice matrices. "
            f"varp={list(adata.varp.keys())}, layers={list(adata.layers.keys())}, "
            f"uns={list(adata.uns.keys())}"
        )


def main():
    args = parse_args()
    nlayers = int(args.nlayers) if args.nlayers > 0 else None
    nhead = int(args.nhead) if args.nhead > 0 else None
    if nlayers is None or nhead is None:
        ckpt_layers, ckpt_heads = load_arch_from_ckpt(args.ckpt)
        nlayers = nlayers or ckpt_layers
        nhead = nhead or ckpt_heads

    if args.print_slice_table:
        print(format_slice_table(nlayers, nhead))
        print(f"\nLast-layer-only indices: {resolve_scprint_head_indices('all', nlayers, nhead, True)}")
        return

    script = Path(args.script_path)
    if not script.exists():
        raise FileNotFoundError(script)

    cmd = [
        sys.executable,
        str(script),
        "--ckpt", args.ckpt,
        "--token-pkl", args.token_pkl,
        "--expr-csv", args.expr_csv,
        "--out-prefix", args.out_prefix,
        "--species", args.species,
        "--cell-type-name", args.cell_type_name,
        "--batch-size", str(args.batch_size),
        "--topk", str(args.topk),
        "--filtration", "none",
        "--head-agg", "none",
        "--preprocess", args.preprocess,
        "--forward-mode", args.forward_mode,
        "--ckpt-gene-emb-offset", str(args.ckpt_gene_emb_offset),
    ]
    print(f"[INFO] Run: {' '.join(cmd)}")
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"scPRINT.py failed with exit code {ret.returncode}")

    h5ad_path = Path(f"{args.out_prefix}_{args.cell_type_name}_grn.h5ad")
    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)
    export_from_h5ad(
        h5ad_path,
        Path(args.out_prefix),
        head_indices=args.head_indices,
        grn_key=args.grn_key,
        gene_col=args.gene_col,
        nlayers=nlayers,
        nhead=nhead,
        last_layer_only=args.last_layer_only,
    )


if __name__ == "__main__":
    main()
