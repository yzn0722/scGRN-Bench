import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified entrypoint for attention TSV export across models."
    )
    p.add_argument(
        "--model",
        required=True,
        choices=["geneformer", "scgpt", "langcell", "sccello", "scprint"],
        help="Model name.",
    )
    p.add_argument(
        "--script-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing model scripts.",
    )
    p.add_argument(
        "model_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the model script. Use '--' before model args.",
    )
    return p.parse_args()


def resolve_script(model: str, script_dir: Path) -> Path:
    mapping = {
        "geneformer": "geneformer_line.py",
        "scgpt": "scgpt_line.py",
        "langcell": "export_heads/export_langcell_heads_tsv.py",
        "sccello": "scCello.py",
        "scprint": "scPRINT.py",
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
