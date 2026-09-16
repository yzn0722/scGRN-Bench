import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified entrypoint for per-head TSV export.")
    p.add_argument(
        "--model",
        required=True,
        choices=["geneformer", "scgpt", "langcell", "sccello", "scprint"],
        help="Model name for per-head export.",
    )
    p.add_argument(
        "--script-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing head-export scripts.",
    )
    p.add_argument(
        "model_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the selected model head-export script. Use '--' before model args.",
    )
    return p.parse_args()


def resolve_script(model: str, script_dir: Path) -> Path:
    mapping = {
        "geneformer": "export_geneformer_heads_tsv.py",
        "scgpt": "export_scgpt_heads_tsv.py",
        "langcell": "export_langcell_heads_tsv.py",
        "sccello": "export_sccello_heads_tsv.py",
        "scprint": "export_scprint_heads_tsv.py",
    }
    script = script_dir / mapping[model]
    if not script.exists():
        raise FileNotFoundError(f"Script not found for model '{model}': {script}")
    return script


def main() -> int:
    args = parse_args()
    script_dir = Path(args.script_dir).resolve()
    script_path = resolve_script(args.model, script_dir)
    passthrough = list(args.model_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    cmd = [sys.executable, str(script_path)] + passthrough
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Script: {script_path}")
    print(f"[INFO] Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
