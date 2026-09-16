import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch
from datasets import load_from_disk
from geneformer import TranscriptomeTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_heads import parse_head_indices


def parse_args():
    p = argparse.ArgumentParser(description="Export scCello per-head TSV files.")
    p.add_argument("--data-type", required=True, choices=["CHIP", "Non_CHIP", "STRING"])
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--input-root", required=True, type=str)
    p.add_argument("--output-root", required=True, type=str)
    p.add_argument("--model-path", required=True, type=str)
    p.add_argument("--dict-dir", required=True, type=str)
    _proj = Path(__file__).resolve().parents[4]  # .../scGRN-Bench
    p.add_argument(
        "--sccello-repo-dir",
        default=str(_proj / "models" / "sc_foundation_evals"),
        type=str,
        help="Path to models/sc_foundation_evals (contains sccello/).",
    )
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--target-layer", default=-1, type=int)
    p.add_argument(
        "--head-indices",
        default="all",
        type=str,
        help="Comma-separated head indices, e.g. '0,1'. Default 'all'.",
    )
    return p.parse_args()


class Collator:
    def __init__(self, add_cls=True):
        self.add_cls = add_cls

    def __call__(self, feats):
        max_len = max(len(f["input_ids"]) for f in feats) + (1 if self.add_cls else 0)
        bid, bmask, bsidx = [], [], []
        for f in feats:
            ids = f["input_ids"]
            sidx = f.get("sorted_indices", list(range(len(ids))))
            if self.add_cls:
                ids = [0] + ids
                sidx = [0] + [x + 1 for x in sidx]
            pad = max_len - len(ids)
            bid.append(ids + [0] * pad)
            bmask.append([1] * len(ids) + [0] * pad)
            bsidx.append(sidx + [0] * pad)
        return {
            "input_ids": torch.tensor(bid),
            "attention_mask": torch.tensor(bmask),
            "sorted_indices": torch.tensor(bsidx),
        }


def reverse_permute(t, idx):
    b = t.shape[0]
    rev = torch.argsort(idx, dim=1)
    if len(t.shape) == 3:
        m = t.shape[1]
        e = rev.unsqueeze(1).expand(-1, m, -1)
        t = torch.gather(t, 2, e)
        e = rev.unsqueeze(2).expand(-1, -1, m)
        t = torch.gather(t, 1, e)
        return t
    if len(t.shape) == 2:
        return torch.gather(t, 1, rev)
    raise ValueError(f"Unsupported shape: {t.shape}")


def main():
    args = parse_args()
    repo_dir = Path(args.sccello_repo_dir)
    if not repo_dir.exists():
        raise FileNotFoundError(f"--sccello-repo-dir does not exist: {repo_dir}")

    # Try multiple possible import roots/layouts for different scCello checkouts.
    candidate_roots = [
        repo_dir,
        repo_dir / "src",
        repo_dir / "sccello",
        repo_dir / "sccello" / "src",
    ]
    for root in candidate_roots:
        if root.exists():
            sys.path.insert(0, str(root))

    scCelloModel = None
    import_errors = []
    for mod_name in [
        "model_prototype_contrastive",
        "sccello.src.model_prototype_contrastive",
        "src.model_prototype_contrastive",
    ]:
        try:
            mod = __import__(mod_name, fromlist=["PrototypeContrastiveForMaskedLM"])
            scCelloModel = getattr(mod, "PrototypeContrastiveForMaskedLM")
            break
        except Exception as e:
            import_errors.append(f"{mod_name}: {e}")

    if scCelloModel is None:
        raise ModuleNotFoundError(
            "Failed to import PrototypeContrastiveForMaskedLM. "
            f"Checked repo={repo_dir}. Tried modules: {import_errors}"
        )

    output_root = Path(args.output_root) / "sccello"
    output_root.mkdir(parents=True, exist_ok=True)

    suffix = "_chip_matched" if args.data_type == "CHIP" else "_processed"
    subdir = "CHIP" if args.data_type == "CHIP" else args.data_type
    csv_path = Path(args.input_root) / subdir / f"{args.dataset}{suffix}-ExpressionData.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    dict_dir = Path(args.dict_dir)
    with open(dict_dir / "gene_name_id_dict.pkl", "rb") as f:
        gene_name_id = pickle.load(f)
    with open(dict_dir / "token_dictionary.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(dict_dir / "gene_median_dictionary.pkl", "rb") as f:
        gene_median_dict = pickle.load(f)

    adata = sc.read_csv(str(csv_path)).T
    gene_symbols = adata.var_names.tolist()
    valid_mask = [g in gene_name_id for g in gene_symbols]
    if sum(valid_mask) == 0:
        raise ValueError("No genes match gene_name_id_dict.pkl; check symbol naming.")
    adata = adata[:, valid_mask]
    final_gene_symbols = [g for g, ok in zip(gene_symbols, valid_mask) if ok]
    ensembl_ids = [gene_name_id[g] for g in final_gene_symbols]
    adata.var["gene_name"] = final_gene_symbols
    adata.var["ensembl_id"] = ensembl_ids
    adata.var_names = final_gene_symbols

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.obs["n_counts"] = adata.X.sum(axis=1).A1 if scipy.sparse.issparse(adata.X) else adata.X.sum(axis=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "unknown"
    adata.obs["adata_order"] = range(adata.n_obs)
    print(f"[INFO] After QC: cells={adata.n_obs}, genes={adata.n_vars}")

    temp_loom = output_root / f"temp_{args.data_type}_{args.dataset}.loom"
    temp_ds = output_root / f"temp_{args.data_type}_{args.dataset}.dataset"
    adata.write_loom(str(temp_loom), write_obsm_varm=False)
    tok = TranscriptomeTokenizer(
        custom_attr_name_dict={"cell_type": "cell_type", "adata_order": "adata_order"},
        nproc=1,
        gene_median_file=str(dict_dir / "gene_median_dictionary.pkl"),
        token_dictionary_file=str(dict_dir / "token_dictionary.pkl"),
    )
    tok.tokenize_data(str(output_root), str(output_root), f"temp_{args.data_type}_{args.dataset}", "loom", use_generator=False)
    tokenized = load_from_disk(str(temp_ds))
    print(f"[INFO] Tokenized cells: {len(tokenized)}")

    def _normalize_example(e):
        input_ids = e["input_ids"] if isinstance(e["input_ids"], list) else e["input_ids"].tolist()
        if "sorted_indices" in e:
            sorted_indices = e["sorted_indices"] if isinstance(e["sorted_indices"], list) else e["sorted_indices"].tolist()
        else:
            sorted_indices = list(range(len(input_ids)))
        return {"input_ids": input_ids, "sorted_indices": sorted_indices}

    tokenized = tokenized.map(_normalize_example, num_proc=1)

    id2name = {v: k for k, v in gene_name_id.items()}
    genelist_dict = {k: True for k in gene_median_dict.keys()}
    coding_loc = np.where([genelist_dict.get(i, False) for i in adata.var["ensembl_id"]])[0]
    ori_gene_names = [id2name.get(i, "") for i in adata.var["ensembl_id"][coding_loc]]
    token_id_to_gene = {}
    for ensembl_id, gene_name in zip(adata.var["ensembl_id"][coding_loc], ori_gene_names):
        token_id = vocab.get(ensembl_id)
        if token_id is not None and gene_name:
            token_id_to_gene[token_id] = gene_name
    print(f"[INFO] token_id_to_gene: {len(token_id_to_gene)}")

    model = scCelloModel.from_pretrained(args.model_path, output_hidden_states=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dl = DataLoader(tokenized, batch_size=int(args.batch_size), shuffle=False, collate_fn=Collator(add_cls=True), num_workers=0)

    per_head = None
    with torch.no_grad():
        for batch in tqdm(dl, desc="Extracting sccello heads"):
            inp = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            sidx = batch["sorted_indices"].to(device)
            out = model.bert(input_ids=inp, attention_mask=am, output_attentions=True)
            attn = out.attentions[int(args.target_layer)]  # [b,h,m,m]
            h = attn.size(1)
            attn = attn[..., 1:, 1:]
            sidx = sidx[:, 1:] - 1
            inp = inp[:, 1:]
            m = attn.shape[-1]
            if m == 0:
                print("[WARN] Skip batch: no gene tokens after CLS removal (empty input_ids?)")
                continue
            attn = attn.reshape((-1, m))
            order = torch.argsort(attn, dim=1)
            rank = torch.argsort(order, dim=1)
            attn = rank.reshape((-1, h, m, m)).float() / float(m)
            attn = attn.permute(0, 1, 3, 2).reshape((-1, m))
            order = torch.argsort(attn, dim=1)
            rank = torch.argsort(order, dim=1)
            attn = (rank.reshape((-1, h, m, m)).float() / float(m)).permute(0, 1, 3, 2)

            bsz = attn.shape[0]
            if per_head is None:
                per_head = [defaultdict(lambda: {"sum": 0.0, "count": 0}) for _ in range(h)]

            for hi in range(h):
                ah = reverse_permute(attn[:, hi], sidx).cpu().numpy()
                gid = reverse_permute(inp, sidx).cpu().numpy()
                for b in range(bsz):
                    names = [token_id_to_gene.get(int(x)) for x in gid[b]]
                    for i in range(len(names)):
                        if names[i] is None:
                            continue
                        for j in range(len(names)):
                            if i == j or names[j] is None:
                                continue
                            key = (names[i], names[j])
                            per_head[hi][key]["sum"] += float(ah[b, i, j])
                            per_head[hi][key]["count"] += 1

    if per_head is None:
        raise RuntimeError("No attention accumulated; check tokenization and gene dictionary mapping.")
    n_heads = len(per_head)
    for hi in parse_head_indices(args.head_indices, n_heads):
        acc = per_head[hi]
        rows = []
        for (g1, g2), st in acc.items():
            if st["count"] > 0:
                rows.append({"Gene1": g1, "Gene2": g2, "EdgeWeight": st["sum"] / st["count"]})
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)
        out = output_root / f"sccello_{args.dataset}_head{hi}.tsv"
        df.to_csv(out, sep="\t", index=False)
        print(f"[INFO] Saved {out} edges={len(df)}")

    try:
        if temp_loom.exists():
            os.remove(temp_loom)
        if temp_ds.exists():
            import shutil
            shutil.rmtree(temp_ds, ignore_errors=True)
    except Exception as e:
        print(f"[WARN] cleanup failed: {e}")


if __name__ == "__main__":
    main()
