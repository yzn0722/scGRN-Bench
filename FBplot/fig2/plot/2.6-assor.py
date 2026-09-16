#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grouped bar chart: Assortativity"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from fig2_palette import model_color

plt.rcParams["font.size"] = 16
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16

ROOT = Path("/mnt/10T/yzn/benchmark_GRN")
EVL_ROOT = ROOT / "evl_omipath"
STRING_ROOT = ROOT / "input_process/STRING"

EXTRACTIONS = ("emb500", "att500", "embhidden500")
EXTRACTION_DISPLAY = {"emb500": r"cos$_{tok}$","att500": "attn","embhidden500": r"cos$_{hid}$"}
MODELS = ("Geneformer", "LangCell", "scGPT", "scCello", "scFoundation", "scPrint")
EXTRACT_COLORS = {"emb500": model_color("scGPT"),"att500": model_color("LangCell"),"embhidden500": model_color("scFoundation")}
MODEL_DIRS = {"Geneformer": {"emb500": "geneformer","att500": "geneformer","embhidden500": "geneformer"},"LangCell": {"emb500": "langcell","att500": "langcell","embhidden500": "Langcell"},"scGPT": {"emb500": "scgpt","att500": "scgpt","embhidden500": "scgpt"},"scCello": {"emb500": "sccello","att500": "sccello","embhidden500": "sccello"},"scFoundation": {"emb500": "scFoundation","att500": "scFoundation","embhidden500": "scFoundation"},"scPrint": {"emb500": "scprint","att500": "scprint","embhidden500": "scprint"}}
MODEL_STEMS = {"Geneformer": ("geneformer","Geneformer"),"LangCell": ("LangCell","langcell","Langcell"),"scGPT": ("scgpt","scGPT","scGPT2"),"scCello": ("scCello","sccello"),"scFoundation": ("scFoundation","scfoundation"),"scPrint": ("scprint","scPrint","scPRINT")}
WEIGHT_COLS = ("EdgeWeight","edgeweight","edge_weight","Attention score","Weight","Score","Importance")

def normalize_edges(df):
    d=df.copy()
    if "Gene1" not in d.columns or "Gene2" not in d.columns:
        if d.shape[1]<2:raise ValueError()
        d=d.iloc[:,:2].copy()
        d.columns=["Gene1","Gene2"]
    d["Gene1"]=d["Gene1"].astype(str).str.strip()
    d["Gene2"]=d["Gene2"].astype(str).str.strip()
    d=d[(d["Gene1"]!="")&(d["Gene2"]!="")&(d["Gene1"]!=d["Gene2"])]
    d=d.drop_duplicates(subset=["Gene1","Gene2"])
    return d

def detect_weight_col(df):
    for c in WEIGHT_COLS:
        if c in df.columns:return c
    return None

def load_string(dataset):
    fp=STRING_ROOT/f"{dataset}_processed-network.csv"
    if not fp.exists():raise FileNotFoundError()
    df=pd.read_csv(fp)
    df=normalize_edges(df)
    g1=set(df["Gene1"].astype(str))
    gu=set(pd.concat([df["Gene1"],df["Gene2"]]).astype(str))
    return g1,gu,len(df)

def resolve_pred_path(model,ext,dataset):
    base=EVL_ROOT/f"output_{ext}"/MODEL_DIRS[model][ext]
    if not base.exists():raise FileNotFoundError()
    for stem in MODEL_STEMS[model]:
        cand=base/f"{stem}_{dataset}.tsv"
        if cand.exists():return cand
    for p in base.glob("*.tsv"):
        if dataset.lower() in p.name.lower():return p
    raise FileNotFoundError()

def load_filtered_pred(fp,g1,gu,top_k):
    df=pd.read_csv(fp,sep="\t")
    df=normalize_edges(df)
    df=df[df["Gene1"].isin(g1)&df["Gene2"].isin(gu)].copy()
    if df.empty:return df
    w=detect_weight_col(df)
    if w:
        df[w]=pd.to_numeric(df[w],errors="coerce").fillna(0)
        df=df.sort_values(w,ascending=False).head(top_k)
    else:
        df=df.head(top_k)
    return df[["Gene1","Gene2"]].copy()

def calc_assort(edges):
    if edges.empty:return np.nan
    g=nx.Graph()
    g.add_edges_from(edges.itertuples(index=False,name=None))
    if g.number_of_nodes()<2:return np.nan
    try:
        return nx.degree_assortativity_coefficient(g)
    except:
        return np.nan

def calc_string_metric(dataset):
    fp=STRING_ROOT/f"{dataset}_processed-network.csv"
    if not fp.exists():return np.nan
    try:
        df=pd.read_csv(fp)
        ed=normalize_edges(df)[["Gene1","Gene2"]].copy()
        return calc_assort(ed)
    except:
        return np.nan

def compute_table(dataset,top_k):
    g1,gu,n_str=load_string(dataset)
    k=n_str if top_k<=0 else min(top_k,n_str)
    rows=[]
    for m in MODELS:
        for e in EXTRACTIONS:
            try:
                fp=resolve_pred_path(m,e,dataset)
                ed=load_filtered_pred(fp,g1,gu,k)
                val=calc_assort(ed)
                rows.append({"Dataset":dataset,"Model":m,"Extraction":e,"Value":val,"Edges":len(ed)})
            except:
                rows.append({"Dataset":dataset,"Model":m,"Extraction":e,"Value":np.nan,"Edges":0})
    return pd.DataFrame(rows)

def plot_bar(df,out,ref_val=np.nan):
    out.parent.mkdir(parents=True, exist_ok=True)
    p=df.pivot(index="Model",columns="Extraction",values="Value").reindex(MODELS,columns=EXTRACTIONS)
    x=np.arange(len(MODELS))
    w=0.24
    fig,ax=plt.subplots(figsize=(6,6),dpi=300)
    for i,e in enumerate(EXTRACTIONS):
        ax.bar(x+(i-1)*w,p[e].values,width=w,color=EXTRACT_COLORS[e],label=EXTRACTION_DISPLAY[e])
    if np.isfinite(ref_val):
        ax.axhline(ref_val,color=model_color("STRING"),linestyle="--",lw=2,zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS,rotation=25,ha="right")
    ax.set_ylabel("Assortativity")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,ncol=5,loc="upper center",bbox_to_anchor=(0.5,1.15))
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"),bbox_inches="tight")
    plt.close()

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--dataset",default="hESC")
    a.add_argument("--top-edges",type=int,default=0)
    args=a.parse_args()

    script_dir = Path(__file__).resolve().parent
    out=script_dir / "output" / "assortativity_bar" / f"{args.dataset}_assortativity_6models.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    df=compute_table(args.dataset,args.top_edges)
    ref=calc_string_metric(args.dataset)

    csv_path = out.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    plot_bar(df,out,ref)
    print("✅ Done:",out)

if __name__=="__main__":main()