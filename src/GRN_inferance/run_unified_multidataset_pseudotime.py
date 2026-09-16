#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launcher: the editable .py source was removed, but the compiled bytecode with
trajectory-probe support remains under __pycache__/.

This wrapper loads:
  __pycache__/run_unified_multidataset_pseudotime.cpython-38.pyc
and forwards to its main().

Supported for --run-trajectory-probes:
  scgpt, scfoundation   (continuous expression-space iteration)

Token models (geneformer / langcell / sccello) can still run the direction
benchmark, but reviewer probes (early/late/mid centroid geometry) are skipped.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path


def _dispatch_ae_control() -> bool:
    """Route generic AE/DAE controls to the lightweight control runner."""
    if "--model" not in sys.argv:
        return False
    i = sys.argv.index("--model")
    if i + 1 >= len(sys.argv):
        return False
    aliases = {
        "ae": "ae",
        "autoencoder": "ae",
        "dae": "dae",
        "denoising_autoencoder": "dae",
    }
    requested = sys.argv[i + 1].lower()
    if requested not in aliases:
        return False

    model_type = aliases[requested]
    argv = sys.argv[:i] + sys.argv[i + 2 :]
    argv += ["--model-type", model_type]

    # Keep the unified runner's outdir/model convention.
    if "--outdir" in argv:
        oi = argv.index("--outdir")
        if oi + 1 < len(argv):
            argv[oi + 1] = str(Path(argv[oi + 1]) / model_type)

    # The control runner accepts one selected dataset; no selector means all.
    if "--datasets" in argv:
        di = argv.index("--datasets")
        selected = argv[di + 1] if di + 1 < len(argv) else ""
        del argv[di : di + 2]
        names = [x.strip() for x in selected.split(",") if x.strip()]
        if len(names) > 1:
            raise ValueError("AE/DAE control currently accepts at most one --datasets entry")
        if names:
            argv += ["--dataset", names[0]]

    # Match scGPT's default preprocessing (no internal log1p during binning).
    if "--use-log1p" in argv:
        argv.remove("--use-log1p")
        argv.append("--log1p")
    if "--no-log1p" in argv:
        argv.remove("--no-log1p")

    sys.argv = argv
    runner = Path(__file__).resolve().parent / "dyn" / "ae_dyn.py"
    print(f"[launcher] generic control model={model_type}; runner={runner}")
    runpy.run_path(str(runner), run_name="__main__")
    return True


if _dispatch_ae_control():
    raise SystemExit(0)


def _load_from_pyc():
    here = Path(__file__).resolve().parent
    # Prefer the 3.8 bytecode (matches singlecell env); fall back to 3.12 if needed.
    candidates = [
        here / "__pycache__" / "run_unified_multidataset_pseudotime.cpython-38.pyc",
        here / "__pycache__" / "run_unified_multidataset_pseudotime.cpython-312.pyc",
    ]
    pyc = next((p for p in candidates if p.is_file()), None)
    if pyc is None:
        raise FileNotFoundError(
            "Missing compiled module under __pycache__/. "
            "Expected run_unified_multidataset_pseudotime.cpython-38.pyc"
        )
    name = "run_unified_multidataset_pseudotime_bytecode"
    spec = importlib.util.spec_from_file_location(name, pyc)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load bytecode: {pyc}")
    mod = importlib.util.module_from_spec(spec)
    # Keep a stable module name for relative imports / pickle if any.
    sys.modules[name] = mod
    sys.modules["run_unified_multidataset_pseudotime"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_from_pyc()


def _patch_scf_gather_data(mod) -> None:
    """
    Fix pad/zero collision in scf_gather_data.

    Bug: padding_labels = new_data.eq(pad_token_id) treats real expression 0.0 as pad
    when pad_token_id==0, so encoder hidden count (6311) != encoder_labels count (6312).

    Match pre_scfoundation/final.py: structural pad = indices from the fake right block.
    """
    import torch

    def scf_gather_data(data: torch.Tensor, labels: torch.Tensor, pad_token_id: int):
        n_genes = int(data.shape[1])
        value_nums = labels.sum(1)
        max_num = int(value_nums.max().item())
        fake_data = torch.full((data.shape[0], max_num), pad_token_id, device=data.device, dtype=data.dtype)
        data2 = torch.hstack([data, fake_data])
        fake_label = torch.full((labels.shape[0], max_num), 1, device=labels.device)
        none_labels = ~labels
        labels_f = labels.float()
        labels_f[none_labels] = torch.tensor(-float("Inf"), device=labels.device)
        tmp_data = torch.tensor(
            [(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)],
            device=labels.device,
            dtype=labels_f.dtype,
        )
        labels_f = labels_f + tmp_data
        labels_f = torch.hstack([labels_f, fake_label])
        idx = labels_f.topk(max_num).indices
        new_data = torch.gather(data2, 1, idx)
        padding_labels = idx >= n_genes
        return new_data, padding_labels

    mod.scf_gather_data = scf_gather_data
    print("[launcher] patched scf_gather_data (structural padding; fix 0==pad collision)")


_patch_scf_gather_data(_mod)

# Re-export common symbols so other scripts can still import from this path.
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})


def _normalize_scf_root_argv():
    """Accept either scFoundation repo root or .../model (parent of pretrainmodels)."""
    if "--scf-root" not in sys.argv:
        return
    i = sys.argv.index("--scf-root")
    if i + 1 >= len(sys.argv):
        return
    root = Path(sys.argv[i + 1]).expanduser().resolve()
    if (root / "pretrainmodels").is_dir():
        return
    if (root / "model" / "pretrainmodels").is_dir():
        sys.argv[i + 1] = str(root / "model")
        print(f"[launcher] --scf-root adjusted to {sys.argv[i + 1]} (contains pretrainmodels/)")


if __name__ == "__main__":
    _normalize_scf_root_argv()
    _mod.main()
