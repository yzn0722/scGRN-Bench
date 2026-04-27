#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import gc
import time
import traceback
import inspect
from types import MethodType

import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

from scprint import scPrint
from scprint.tasks.grn import GNInfer


# ===================
# CLI 参数（禁止硬编码绝对路径）
# ===================
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Run scPRINT GNInfer on one ExpressionData.csv and write a GRN .h5ad.")
    p.add_argument("--ckpt", required=True, type=str, help="Path to scPRINT .ckpt file.")
    p.add_argument("--token-pkl", required=True, type=str, help="Path to token_dictionary.pkl.")
    p.add_argument("--expr-csv", required=True, type=str, help="Path to ExpressionData.csv (genes x cells).")
    p.add_argument("--out-prefix", required=True, type=str, help="Output prefix (directory will be created).")
    p.add_argument("--species", default="NCBITaxon:9606", type=str, help="Species ontology term id.")
    p.add_argument("--cell-type-name", default="all", type=str, help="Cell type label stored in adata.obs.")
    p.add_argument("--batch-size", default=64, type=int, help="GNInfer batch size.")
    p.add_argument("--topk", default=10, type=int, help="TopK edges per gene (when filtration uses top-k).")
    p.add_argument("--filtration", default="none", choices=["top-k", "thresh", "none"], help="Edge filtration.")
    p.add_argument("--head-agg", default="mean", choices=["mean", "max", "none"], help="Attention head aggregation.")
    p.add_argument("--preprocess", default="softmax", choices=["softmax", "sinkhorn", "none"], help="Preprocess.")
    p.add_argument("--forward-mode", default="none", type=str, help="Forward mode (keep 'none' unless needed).")
    p.add_argument("--ckpt-gene-emb-offset", default=0, type=int, help="Embedding row offset (if ckpt has specials).")
    return p.parse_args()


ARGS = parse_args()

CKPT = ARGS.ckpt
TOKEN_PKL = ARGS.token_pkl
CSV_GENE_BY_CELL = ARGS.expr_csv
OUT_PREFIX = ARGS.out_prefix
SPECIES = ARGS.species
CELL_TYPE_NAME = ARGS.cell_type_name

# 推断参数（可改）
BATCH_SIZE = int(ARGS.batch_size)
TOPK = int(ARGS.topk)
FILTRATION = str(ARGS.filtration)     # "top-k" / "thresh" / "none"
HEAD_AGG = str(ARGS.head_agg)         # "mean" / "max" / "none"
PREPROCESS = str(ARGS.preprocess)     # "softmax" / "sinkhorn" / "none"
FORWARD_MODE = str(ARGS.forward_mode) # 一般保持 none

# 如果你发现 ckpt 的 gene embedding 行号相对 ckpt_genes 有偏移（比如 embedding 前面还有 special token），可调这个
CKPT_GENE_EMB_OFFSET = int(ARGS.ckpt_gene_emb_offset)


def write_txt(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
        f.flush()
        os.fsync(f.fileno())


def datetime_now():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_state_dict_keys(state_dict: dict):
    prefixes = ["model.", "scprint.", "net.", "module."]
    out = {}
    for k, v in state_dict.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
                break
        out[kk] = v
    return out


def _is_ensg(x: str) -> bool:
    return isinstance(x, str) and x.startswith("ENSG")


def load_env_genes_from_token_pkl(token_pkl: str):
    """token_dictionary.pkl: dict(token->id). itos[0]=<pad>, itos[1]=<mask>"""
    import pickle

    with open(token_pkl, "rb") as f:
        stoi = pickle.load(f)
    if not isinstance(stoi, dict):
        raise TypeError(f"Expected dict token->id, got: {type(stoi)}")

    max_id = max(stoi.values())
    itos = [None] * (max_id + 1)
    for tok, idx in stoi.items():
        itos[idx] = tok
    if any(x is None for x in itos):
        raise RuntimeError("itos has None: ids not contiguous or missing.")

    if len(itos) >= 2 and itos[0] == "<pad>" and itos[1] == "<mask>":
        genes = itos[2:]
    else:
        special = {"<pad>", "<mask>", "<unk>", "<cls>", "<bos>", "<eos>"}
        genes = [t for t in itos if isinstance(t, str) and t not in special]

    ensg_ratio = sum(_is_ensg(g) for g in genes) / max(1, len(genes))
    if ensg_ratio < 0.5:
        raise RuntimeError(f"Loaded genes do not look like ENSG tokens (ENSG ratio={ensg_ratio:.3f}).")

    return genes, itos


def clean_symbol(sym: str) -> str:
    s = str(sym).strip()
    s = re.sub(r"\s+", "", s)
    return s


def map_symbols_to_ensembl(symbols, species_taxon: str, target_ens_set: set, log_path: str):
    try:
        from mygene import MyGeneInfo
    except Exception as e:
        raise RuntimeError(
            "缺少依赖 mygene。请先运行：pip install mygene\n"
            f"import error: {e}"
        )

    mg = MyGeneInfo()
    sp_name = "human" if species_taxon == "NCBITaxon:9606" else "mouse"

    res = mg.querymany(
        [clean_symbol(s) for s in symbols],
        scopes="symbol",
        fields="ensembl.gene",
        species=sp_name,
        as_dataframe=True,
        returnall=False,
        verbose=False,
    )

    mapping = {}
    for sym, row in res.iterrows():
        val = None
        try:
            val = row.get("ensembl.gene", None)
        except Exception:
            val = None

        ens = None
        if isinstance(val, dict) and "gene" in val:
            ens = val["gene"]
        elif isinstance(val, list) and len(val) > 0:
            v0 = val[0]
            if isinstance(v0, dict) and "gene" in v0:
                ens = v0["gene"]
            elif isinstance(v0, str):
                ens = v0
        elif isinstance(val, str):
            ens = val

        if ens is not None and ens in target_ens_set:
            mapping[str(sym)] = ens

    write_txt(log_path, "=== Gene mapping report ===")
    write_txt(log_path, f"symbols_in: {len(symbols)}")
    write_txt(log_path, f"mapped_and_in_model: {len(mapping)}")
    return mapping


def csv_to_adata(csv_path: str, model_gene_set: set, log_path: str):
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    mapping = map_symbols_to_ensembl(df.index.tolist(), SPECIES, model_gene_set, log_path)
    if len(mapping) == 0:
        raise RuntimeError("symbol->Ensembl 映射后为 0，请检查 CSV 行名是否为 gene symbol。")

    df2 = df.loc[list(mapping.keys())].copy()
    df2.index = [mapping[s] for s in df2.index]
    df2 = df2.groupby(df2.index).sum()

    X = df2.T.values
    sparsity = float((X == 0).mean())
    if sparsity > 0.7:
        X = sp.csr_matrix(X)

    adata = sc.AnnData(X=X)
    adata.obs_names = df2.columns
    adata.var_names = df2.index

    adata.obs["organism_ontology_term_id"] = SPECIES
    adata.obs["cell_type"] = CELL_TYPE_NAME

    inv = {}
    for sym, ens in mapping.items():
        inv.setdefault(ens, sym)
    adata.var["symbol"] = [inv.get(ens, ens) for ens in adata.var_names]

    write_txt(log_path, "=== Data report ===")
    write_txt(log_path, f"csv: {csv_path}")
    write_txt(log_path, f"raw df shape (genes×cells): {df.shape}")
    write_txt(log_path, f"after mapping df shape (ensembl_genes×cells): {df2.shape}")
    write_txt(log_path, f"adata shape (cells×genes): {adata.shape}")
    write_txt(log_path, f"sparsity approx: {sparsity:.3f}")
    write_txt(log_path, "adata.var columns: " + ",".join(list(adata.var.columns)))
    return adata


def disable_bias_runtime(model, log_path=None):
    model.attn_bias = "none"
    if hasattr(model, "nbias"):
        try:
            delattr(model, "nbias")
        except Exception:
            pass

    def new_forward(
        self,
        gene_pos,
        expression=None,
        mask=None,
        req_depth=None,
        timepoint=None,
        get_gene_emb=False,
        metacell_token=None,
        depth_mult=None,
        do_sample=False,
        do_mvc=False,
        do_class=False,
        get_attention_layer=None,
    ):
        if get_attention_layer is None:
            get_attention_layer = []

        encoding = self._encoder(
            gene_pos,
            expression,
            mask,
            req_depth=req_depth if self.depth_atinput else None,
            timepoint=timepoint,
            metacell_token=metacell_token,
        )

        if self.cell_transformer:
            cell_encoding = encoding[:, : self.cell_embs_count, :]
            encoding = encoding[:, self.cell_embs_count :, :]

        transformer_output = self.transformer(
            encoding,
            return_qkv=get_attention_layer,
            bias=None,
            bias_layer=list(range(self.nlayers - 1)),
        )

        if len(get_attention_layer) > 0:
            transformer_output, qkvs = transformer_output

        if self.cell_transformer:
            cell_output = self.cell_transformer(cell_encoding, x_kv=transformer_output)
            transformer_output = torch.cat([cell_output, transformer_output], dim=1)

        depth_mult2 = expression.sum(1) if depth_mult is None else depth_mult
        res = self._decoder(
            transformer_output,
            depth_mult2,
            get_gene_emb,
            do_sample,
            do_mvc,
            do_class,
            req_depth=req_depth if not self.depth_atinput else None,
        )
        return (res, qkvs) if len(get_attention_layer) > 0 else res

    model.forward = MethodType(new_forward, model)
    if log_path:
        write_txt(log_path, "Bias disabled: attn_bias='none' + forward monkey-patched.")


def load_model_with_embedding_alignment_fp32_normal(ckpt_path: str, token_pkl: str, log_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_genes, env_itos = load_env_genes_from_token_pkl(token_pkl)
    write_txt(log_path, f"START: {datetime_now()}")
    write_txt(log_path, f"env_itos_len(including specials): {len(env_itos)}")
    write_txt(log_path, f"env_genes_len: {len(env_genes)}")
    write_txt(log_path, f"env_itos_head10: {env_itos[:10]}")
    write_txt(log_path, f"env_genes_head10: {env_genes[:10]}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})
    state = ckpt.get("state_dict", ckpt)
    if isinstance(state, dict):
        state = _normalize_state_dict_keys(state)

    ckpt_genes = hp.get("genes", None)
    if not isinstance(ckpt_genes, (list, tuple)) or len(ckpt_genes) == 0:
        raise RuntimeError("ckpt hyper_parameters 里没有有效 genes 列表，无法对齐 embedding。")
    write_txt(log_path, f"ckpt_genes_len: {len(ckpt_genes)}")

    emb_key = "gene_encoder.embeddings.weight"
    ckpt_emb = state.get(emb_key, None)
    if not torch.is_tensor(ckpt_emb):
        raise RuntimeError(f"state_dict 里找不到 {emb_key}。")
    write_txt(log_path, f"ckpt {emb_key} shape: {tuple(ckpt_emb.shape)}")

    sig = inspect.signature(scPrint.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    kwargs = {k: v for k, v in hp.items() if k in allowed}

    kwargs.pop("genes", None)
    kwargs.pop("nb_features", None)
    kwargs["genes"] = env_genes

    if "transformer" in allowed:
        kwargs["transformer"] = "normal"
    if "attn_bias" in allowed:
        kwargs["attn_bias"] = "none"
    if "precpt_gene_emb" in allowed:
        kwargs["precpt_gene_emb"] = None

    model = scPrint(**kwargs)
    model = model.to(device).to(torch.float32)
    model.eval()

    model_sd = model.state_dict()
    mw = model_sd.get(emb_key, None)
    if not torch.is_tensor(mw):
        raise RuntimeError(f"模型里没有 {emb_key}。")

    model_genes = list(getattr(model, "genes", []))
    write_txt(log_path, f"model_genes_len: {len(model_genes)}")
    write_txt(log_path, f"model {emb_key} shape: {tuple(mw.shape)}")

    if mw.shape[1] != ckpt_emb.shape[1]:
        raise RuntimeError(f"embedding dim mismatch: ckpt={tuple(ckpt_emb.shape)} model={tuple(mw.shape)}")

    ckpt_idx = {g: i for i, g in enumerate(ckpt_genes)}
    model_to_ckpt_row = [ckpt_idx.get(g, -1) for g in model_genes]
    hit = sum(r >= 0 for r in model_to_ckpt_row)
    miss = len(model_genes) - hit
    write_txt(log_path, f"overlap(hit): {hit}/{len(model_genes)} ({hit/len(model_genes):.3%})")
    write_txt(log_path, f"missing: {miss}/{len(model_genes)} ({miss/len(model_genes):.3%})")

    new_emb = mw.clone()
    copied = 0
    for j, src in enumerate(model_to_ckpt_row):
        if src >= 0:
            src2 = src + int(CKPT_GENE_EMB_OFFSET)
            if 0 <= src2 < ckpt_emb.shape[0]:
                new_emb[j] = ckpt_emb[src2]
                copied += 1
    write_txt(log_path, f"aligned embedding rows copied: {copied}/{len(model_genes)}")

    state[emb_key] = new_emb

    filtered = {}
    skipped = []
    for k, v in state.items():
        if k not in model_sd:
            continue
        if not torch.is_tensor(v):
            continue
        if tuple(v.shape) == tuple(model_sd[k].shape):
            filtered[k] = v
        else:
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    write_txt(log_path, "=== Model load report (fp32 + normal + emb-aligned) ===")
    write_txt(log_path, f"device: {device}")
    write_txt(log_path, f"loaded_keys: {len(filtered)}")
    write_txt(log_path, f"skipped_shape_mismatch: {len(skipped)}")
    if skipped[:5]:
        write_txt(log_path, f"skipped_examples_top5: {skipped[:5]}")
    write_txt(log_path, f"missing_keys_count: {len(missing)}")
    write_txt(log_path, f"unexpected_keys_count: {len(unexpected)}")
    write_txt(log_path, f"organisms: {getattr(model, 'organisms', None)}")
    write_txt(log_path, f"num_genes(model.genes): {len(getattr(model, 'genes', []))}")

    disable_bias_runtime(model, log_path)
    return model


@torch.no_grad()
def infer_grn_fp32_noamp_slice_attn(model, adata, grn_inferer: GNInfer, log_path: str):
    """
    ✅ 关键修复：attn.get() 是全词表 cache，未参与 token 位置用 NaN 占位
    => 把 attn 切到 [cell_tokens + used_genes] 再 aggregate
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_genes = list(model.genes)
    gene_to_idx = {g: i for i, g in enumerate(model_genes)}

    used_genes = [g for g in adata.var_names if g in gene_to_idx]
    if len(used_genes) == 0:
        raise RuntimeError("adata.var_names 与 model.genes 没有交集（确认已是 ENSG）。")

    # 对齐 inferer
    grn_inferer.curr_genes = used_genes
    grn_inferer.genes = used_genes
    grn_inferer.num_genes = len(used_genes)

    gene_pos_1 = torch.tensor([gene_to_idx[g] for g in used_genes], dtype=torch.long, device=device)

    X = adata[:, used_genes].X
    if sp.issparse(X):
        X = X.toarray()
    expression = torch.tensor(X, dtype=torch.float32, device=device)
    depth = expression.sum(dim=1).clamp_min(1e-6)

    if grn_inferer.layer is None:
        grn_inferer.layer = list(range(model.nlayers))
    grn_inferer.n_cell_embs = model.attn.additional_tokens

    model.pred_log_adata = False
    model.eval()
    model.on_predict_epoch_start()

    batch_size = grn_inferer.batch_size
    n_cells = expression.shape[0]

    write_txt(log_path, "=== Running GRN (fp32, no AMP, slice attn cache) ===")
    write_txt(log_path, f"used_genes: {len(used_genes)}")
    write_txt(log_path, f"cells: {n_cells}, batch_size: {batch_size}")

    layers = grn_inferer.layer if isinstance(grn_inferer.layer, list) else [grn_inferer.layer]

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        expr_b = expression[start:end]
        depth_b = depth[start:end]
        gene_pos_b = gene_pos_1.unsqueeze(0).expand(expr_b.shape[0], -1)

        model._predict(
            gene_pos_b,
            expr_b,
            depth_b,
            predict_mode=grn_inferer.forward_mode,
            keep_output=False,
            get_attention_layer=layers,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    attn = model.attn.get()
    if attn is None:
        raise RuntimeError("model.attn.get() returned None (attention not captured).")

    write_txt(log_path, f"attn(raw) shape: {tuple(attn.shape)} dtype: {attn.dtype} device: {attn.device}")
    write_txt(log_path, f"attn(raw) nan: {torch.isnan(attn).sum().item()} inf: {torch.isinf(attn).sum().item()}")

    # =========================
    # ✅ 核心：切 attn 到“实际用到的 token”
    # token 0..n_cell_embs-1: cell special tokens
    # token n_cell_embs + gene_idx: gene token
    # =========================
    n_cell = int(grn_inferer.n_cell_embs)
    gene_ids = torch.tensor([gene_to_idx[g] for g in used_genes], dtype=torch.long, device=attn.device)
    idx_keep = torch.cat(
        [
            torch.arange(n_cell, device=attn.device, dtype=torch.long),
            gene_ids + n_cell,
        ],
        dim=0,
    )

    attn = attn.index_select(1, idx_keep)

    write_txt(log_path, f"attn(sliced) shape: {tuple(attn.shape)}")
    write_txt(log_path, f"attn(sliced) nan: {torch.isnan(attn).sum().item()} inf: {torch.isinf(attn).sum().item()}")

    # ✅ aggregate 用 sliced attn + used_genes（顺序一致）
    adj_all = grn_inferer.aggregate(attn, used_genes)

    if grn_inferer.head_agg == "none":
        grn = grn_inferer.save(
            adj_all[n_cell:, n_cell:, :],
            adata[:, used_genes].copy(),
        )
    else:
        adj = grn_inferer.filter(adj_all)
        grn = grn_inferer.save(
            adj[n_cell:, n_cell:],
            adata[:, used_genes].copy(),
        )

    return grn


def main():
    os.makedirs(os.path.dirname(OUT_PREFIX), exist_ok=True)
    log_path = f"{OUT_PREFIX}_runlog.txt"
    if os.path.exists(log_path):
        os.remove(log_path)

    try:
        t0 = time.time()

        model = load_model_with_embedding_alignment_fp32_normal(CKPT, TOKEN_PKL, log_path)
        model_gene_set = set(model.genes)

        adata = csv_to_adata(CSV_GENE_BY_CELL, model_gene_set, log_path)

        used_genes = [g for g in adata.var_names if g in model_gene_set]
        if len(used_genes) == 0:
            raise RuntimeError("映射后 adata.var_names 与 model.genes 无交集。")
        write_txt(log_path, f"final used_genes: {len(used_genes)}")

        grn_inferer = GNInfer(
            layer=None,
            how="given",
            genes=used_genes,
            num_genes=len(used_genes),
            preprocess=PREPROCESS,
            head_agg=HEAD_AGG,
            filtration=FILTRATION,
            k=TOPK,
            doplot=False,
            batch_size=BATCH_SIZE,
            num_workers=0,
            max_cells=0,
            forward_mode=FORWARD_MODE,
            comp_attn=True,
            dtype=torch.float32,
        )

        grn = infer_grn_fp32_noamp_slice_attn(model, adata, grn_inferer, log_path)

        out_h5ad = f"{OUT_PREFIX}_{CELL_TYPE_NAME}_grn.h5ad"
        grn.write_h5ad(out_h5ad)
        write_txt(log_path, f"Saved GRN: {out_h5ad}")

        del grn, adata, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        write_txt(log_path, f"DONE in {time.time() - t0:.1f}s")
        write_txt(log_path, f"END: {datetime_now()}")

    except Exception:
        write_txt(log_path, "FAILED\n" + traceback.format_exc())


if __name__ == "__main__":
    main()
