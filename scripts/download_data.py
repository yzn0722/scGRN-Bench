#!/usr/bin/env python3
"""One-click download for scGRN-Bench raw datasets.

Downloads and places files under ``data/`` to match this repository's layout:

  data/row_data/omnipath.parquet
  data/Groundtruth/{CHIP,NonCHIP,STRING}/...
  data/row_data/scRNA-Seq/{dataset}/{ExpressionData,GeneOrdering,PseudoTime}.csv
  data/PseudoTime/{dataset}/PseudoTime.csv
  data/process_data/process.py          (BEELINE generateExpInputs.py)
  data/process_data/{human,mouse}-tfs.csv

Sources (see data/readme.txt):
  - omnipath.parquet : scPRINT GitHub data/main
  - Ground truth     : Zenodo 3701939  BEELINE-Networks.zip
  - scRNA-Seq        : Zenodo 3701939  BEELINE-data.zip
  - process.py / TFs : murali-group/BEELINE

Usage:
  python scripts/download_data.py
  python scripts/download_data.py --force
  python scripts/download_data.py --skip-scrna   # skip ~250 MB BEELINE-data.zip
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

OMNIPATH_URL = (
    "https://raw.githubusercontent.com/cantinilab/scPRINT/main/data/main/omnipath.parquet"
)
ZENODO_NETWORKS_URL = (
    "https://zenodo.org/api/records/3701939/files/BEELINE-Networks.zip/content"
)
ZENODO_DATA_URL = (
    "https://zenodo.org/api/records/3701939/files/BEELINE-data.zip/content"
)
PROCESS_PY_URL = (
    "https://raw.githubusercontent.com/murali-group/BEELINE/master/utils/generateExpInputs.py"
)

# Expected MD5 from Zenodo record 3701939 metadata (optional integrity check).
ZENODO_MD5 = {
    "BEELINE-Networks.zip": "a7dca84d302bb127bad962ae52815662",
    "BEELINE-data.zip": "52581e7e27aa699310384238daac4751",
}

CHIP_NETWORK_FILES = [
    "Networks/human/hESC-ChIP-seq-network.csv",
    "Networks/human/HepG2-ChIP-seq-network.csv",
    "Networks/mouse/mDC-ChIP-seq-network.csv",
    "Networks/mouse/mESC-ChIP-seq-network.csv",
    "Networks/mouse/mESC-lofgof-network.csv",
    "Networks/mouse/mHSC-ChIP-seq-network.csv",
]

NONCHIP_NETWORK_FILES = [
    "Networks/human/Non-specific-ChIP-seq-network.csv",
    "Networks/mouse/Non-Specific-ChIP-seq-network.csv",
]

STRING_NETWORK_MAP = {
    "Networks/human/STRING-network.csv": "human_STRING-network.csv",
    "Networks/mouse/STRING-network.csv": "mouse_STRING-network.csv",
}

TF_FILES = ["human-tfs.csv", "mouse-tfs.csv"]

SCRNA_DATASETS = [
    "hESC",
    "hHep",
    "mDC",
    "mESC",
    "mHSC-E",
    "mHSC-GM",
    "mHSC-L",
]


def _human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.1f}{u}" if u != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}GB"


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` with a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  [skip] already exists: {dest}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"  [get ] {url}")
    print(f"  [to  ] {dest}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "scGRN-Bench-download/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = resp.headers.get("Content-Length")
            total_i = int(total) if total and total.isdigit() else None
            downloaded = 0
            chunk = 1024 * 1024
            with open(tmp, "wb") as fh:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    downloaded += len(buf)
                    if total_i:
                        pct = 100.0 * downloaded / total_i
                        msg = (
                            f"\r  [prog] {_human_size(downloaded)} / "
                            f"{_human_size(total_i)} ({pct:5.1f}%)"
                        )
                    else:
                        msg = f"\r  [prog] {_human_size(downloaded)}"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
            sys.stdout.write("\n")
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    tmp.replace(dest)
    print(f"  [ok  ] {_human_size(dest.stat().st_size)} -> {dest}")
    return dest


def md5sum(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def verify_md5(path: Path, expected: str | None) -> None:
    if not expected:
        return
    got = md5sum(path)
    if got != expected:
        raise RuntimeError(
            f"MD5 mismatch for {path.name}: expected {expected}, got {got}"
        )
    print(f"  [md5 ] {path.name} OK")


def extract_member(zf: zipfile.ZipFile, member: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)


def organize_networks(zip_path: Path, force: bool = False) -> None:
    chip_dir = DATA_ROOT / "Groundtruth" / "CHIP"
    nonchip_dir = DATA_ROOT / "Groundtruth" / "NonCHIP"
    string_dir = DATA_ROOT / "Groundtruth" / "STRING"
    process_dir = DATA_ROOT / "process_data"

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

        for member in CHIP_NETWORK_FILES:
            if member not in names:
                print(f"  [warn] missing in zip: {member}")
                continue
            dest = chip_dir / Path(member).name
            if dest.exists() and not force:
                print(f"  [skip] {dest}")
            else:
                extract_member(zf, member, dest)
                print(f"  [put ] {dest}")

        for member in NONCHIP_NETWORK_FILES:
            if member not in names:
                print(f"  [warn] missing in zip: {member}")
                continue
            dest = nonchip_dir / Path(member).name
            if dest.exists() and not force:
                print(f"  [skip] {dest}")
            else:
                extract_member(zf, member, dest)
                print(f"  [put ] {dest}")

        for member, out_name in STRING_NETWORK_MAP.items():
            if member not in names:
                print(f"  [warn] missing in zip: {member}")
                continue
            dest = string_dir / out_name
            if dest.exists() and not force:
                print(f"  [skip] {dest}")
            else:
                extract_member(zf, member, dest)
                print(f"  [put ] {dest}")

        for name in TF_FILES:
            if name not in names:
                print(f"  [warn] missing in zip: {name}")
                continue
            dest = process_dir / name
            if dest.exists() and not force:
                print(f"  [skip] {dest}")
            else:
                extract_member(zf, name, dest)
                print(f"  [put ] {dest}")


def _find_scrna_prefix(names: list[str]) -> str:
    """Return zip prefix that contains inputs/scRNA-Seq/ (handles nested roots)."""
    marker = "inputs/scRNA-Seq/"
    for name in names:
        if marker in name and name.endswith("ExpressionData.csv"):
            idx = name.index(marker)
            return name[:idx]
    raise RuntimeError(
        "Could not locate inputs/scRNA-Seq/ inside BEELINE-data.zip"
    )


def organize_scrna(zip_path: Path, force: bool = False) -> None:
    scrna_root = DATA_ROOT / "row_data" / "scRNA-Seq"
    pt_root = DATA_ROOT / "PseudoTime"

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        prefix = _find_scrna_prefix(names)
        print(f"  [info] zip scRNA prefix: {prefix!r}")

        for dataset in SCRNA_DATASETS:
            base = f"{prefix}inputs/scRNA-Seq/{dataset}/"
            for fname in ("ExpressionData.csv", "GeneOrdering.csv", "PseudoTime.csv"):
                member = base + fname
                if member not in names:
                    # Some zip tools normalize paths; try without duplicate slash
                    alt = member.replace("//", "/")
                    if alt in names:
                        member = alt
                    else:
                        print(f"  [warn] missing: {member}")
                        continue

                dest = scrna_root / dataset / fname
                if dest.exists() and not force:
                    print(f"  [skip] {dest}")
                else:
                    extract_member(zf, member, dest)
                    print(f"  [put ] {dest}")

                if fname == "PseudoTime.csv":
                    pt_dest = pt_root / dataset / "PseudoTime.csv"
                    if pt_dest.exists() and not force:
                        print(f"  [skip] {pt_dest}")
                    else:
                        pt_dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(dest, pt_dest)
                        print(f"  [put ] {pt_dest}")


def download_process_py(force: bool = False) -> None:
    dest = DATA_ROOT / "process_data" / "process.py"
    download_file(PROCESS_PY_URL, dest, force=force)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download scGRN-Bench datasets into data/."
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Destination data root (default: <repo>/data)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Keep downloaded zips here (default: system temp, deleted after).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files.")
    p.add_argument("--skip-omnipath", action="store_true")
    p.add_argument("--skip-networks", action="store_true")
    p.add_argument("--skip-scrna", action="store_true", help="Skip BEELINE-data.zip (~250MB).")
    p.add_argument("--skip-process", action="store_true")
    p.add_argument(
        "--no-verify-md5",
        action="store_true",
        help="Do not verify Zenodo zip MD5 checksums.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    global DATA_ROOT
    DATA_ROOT = args.data_root.resolve()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"scGRN-Bench data root: {DATA_ROOT}")
    keep_cache = args.cache_dir is not None
    cache = (args.cache_dir or Path(tempfile.mkdtemp(prefix="scgrn_dl_"))).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Download cache: {cache}")

    try:
        if not args.skip_omnipath:
            print("\n==> 1/4 omnipath.parquet (scPRINT)")
            download_file(
                OMNIPATH_URL,
                DATA_ROOT / "row_data" / "omnipath.parquet",
                force=args.force,
            )

        if not args.skip_networks:
            print("\n==> 2/4 BEELINE-Networks.zip (Zenodo 3701939)")
            net_zip = cache / "BEELINE-Networks.zip"
            download_file(ZENODO_NETWORKS_URL, net_zip, force=args.force)
            if not args.no_verify_md5:
                verify_md5(net_zip, ZENODO_MD5["BEELINE-Networks.zip"])
            print("  organizing Groundtruth + TF lists ...")
            organize_networks(net_zip, force=args.force)

        if not args.skip_scrna:
            print("\n==> 3/4 BEELINE-data.zip (Zenodo 3701939, ~250MB)")
            data_zip = cache / "BEELINE-data.zip"
            download_file(ZENODO_DATA_URL, data_zip, force=args.force)
            if not args.no_verify_md5:
                verify_md5(data_zip, ZENODO_MD5["BEELINE-data.zip"])
            print("  organizing scRNA-Seq + PseudoTime ...")
            organize_scrna(data_zip, force=args.force)

        if not args.skip_process:
            print("\n==> 4/4 process.py (BEELINE generateExpInputs.py)")
            download_process_py(force=args.force)

        print("\nDone. Expected layout:")
        print(
            f"""
{DATA_ROOT}/
├── row_data/
│   ├── omnipath.parquet
│   └── scRNA-Seq/{{hESC,hHep,...}}/{{ExpressionData,GeneOrdering,PseudoTime}}.csv
├── Groundtruth/
│   ├── CHIP/          # ChIP-seq (+ mESC-lofgof) networks
│   ├── NonCHIP/       # Non-specific ChIP networks
│   └── STRING/        # human_STRING-network.csv, mouse_STRING-network.csv
├── PseudoTime/{{dataset}}/PseudoTime.csv
└── process_data/
    ├── process.py
    ├── human-tfs.csv
    └── mouse-tfs.csv

Note: processed matrices under data/input_process/ (CHIP / Non_CHIP / STRING)
are produced by BEELINE-style preprocessing (process.py), not by this script.
"""
        )
        return 0
    finally:
        if not keep_cache:
            shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
