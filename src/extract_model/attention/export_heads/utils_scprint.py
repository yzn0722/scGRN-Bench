"""scPRINT varp['GRN'] slice indexing: K = nlayers * nhead."""

from typing import List, Tuple

from utils_heads import parse_head_indices


def load_arch_from_ckpt(ckpt_path: str) -> Tuple[int, int]:
    import torch

    hp = torch.load(ckpt_path, map_location="cpu").get("hyper_parameters", {})
    nlayers = int(hp.get("nlayers", 8))
    nhead = int(hp.get("nhead", 4))
    return nlayers, nhead


def slice_index(layer: int, head: int, nhead: int) -> int:
    """Map (layer, head) to index on varp['GRN'] last axis."""
    return layer * nhead + head


def last_layer_head_indices(nlayers: int, nhead: int) -> List[int]:
    """Indices for the final transformer layer (comparable to scCello's 4 heads)."""
    last = nlayers - 1
    return [slice_index(last, h, nhead) for h in range(nhead)]


def resolve_scprint_head_indices(
    spec: str,
    nlayers: int,
    nhead: int,
    last_layer_only: bool = False,
) -> List[int]:
    n_slices = nlayers * nhead
    if last_layer_only:
        return last_layer_head_indices(nlayers, nhead)
    return parse_head_indices(spec, n_slices)


def format_slice_table(nlayers: int, nhead: int) -> str:
    lines = [
        "layer | head | slice_index",
        "------|------|------------",
    ]
    for layer in range(nlayers):
        for head in range(nhead):
            idx = slice_index(layer, head, nhead)
            mark = "  <- last layer" if layer == nlayers - 1 else ""
            lines.append(f"  {layer:2d}  |  {head}   |     {idx:2d}{mark}")
    return "\n".join(lines)
