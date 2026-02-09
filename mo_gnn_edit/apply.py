from __future__ import annotations

import json
from typing import Any

from .chem_utils import (
    atom_idx_by_map,
    ensure_atom_maps,
    mol_from_smiles,
    mol_to_tagged_smiles,
    normalize_bracket_atom_hydrogens,
)
from .edits import (
    attach_functional_group,
    remove_non_scaffold_branch,
    remove_terminal_substituent,
    replace_non_scaffold_branch,
    replace_terminal_substituent,
)


def apply_edits(start_smiles_tagged: str, edits: list[dict[str, Any]]) -> str:
    """
    Apply a list of one-step edits to a tagged SMILES (atoms must have map numbers).

    Expected edit dict fields (same as `Edit.to_dict()`):
      - anchor_atom_map: int
      - op: "add" | "remove" | "replace"
      - fg_smiles: str (required for add/replace)
      - removed_atom_map: int (optional, used to deterministically remove)
    """
    mol = ensure_atom_maps(mol_from_smiles(start_smiles_tagged))
    mol = normalize_bracket_atom_hydrogens(mol)
    for e in edits:
        op = e["op"]
        anchor_map = int(e["anchor_atom_map"])
        anchor_idx = atom_idx_by_map(mol, anchor_map)

        if op == "add":
            fg_smiles = str(e["fg_smiles"])
            mol = attach_functional_group(mol, anchor_idx, fg_smiles)
        elif op == "remove":
            removed_map = e.get("removed_atom_map")
            if removed_map is not None:
                try:
                    mol, _ = remove_non_scaffold_branch(mol, anchor_map, int(removed_map))
                except Exception:
                    mol, _ = remove_terminal_substituent(
                        mol,
                        anchor_map,
                        removed_atom_map=int(removed_map),
                    )
            else:
                mol, _ = remove_terminal_substituent(mol, anchor_map, removed_atom_map=None)
        elif op == "replace":
            fg_smiles = str(e["fg_smiles"])
            removed_map = e.get("removed_atom_map")
            if removed_map is not None:
                try:
                    mol, _ = replace_non_scaffold_branch(mol, anchor_map, int(removed_map), fg_smiles)
                except Exception:
                    mol2, _ = remove_terminal_substituent(
                        mol,
                        anchor_map,
                        removed_atom_map=int(removed_map),
                    )
                    anchor_idx2 = atom_idx_by_map(mol2, anchor_map)
                    mol = attach_functional_group(mol2, anchor_idx2, fg_smiles)
            else:
                mol, _ = replace_terminal_substituent(mol, anchor_map, fg_smiles)
        else:
            raise ValueError(f"Unknown op: {op}")

    return mol_to_tagged_smiles(mol)


def apply_edits_json(start_smiles_tagged: str, edits_json: str) -> str:
    return apply_edits(start_smiles_tagged, json.loads(edits_json))
