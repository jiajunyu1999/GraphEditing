from __future__ import annotations

from typing import Set

import torch


def scaffold_atom_indices(mol: "Chem.Mol") -> set[int]:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return set()
    match = mol.GetSubstructMatch(scaf)
    if not match:
        return set()
    return set(int(i) for i in match)


def scaffold_mask(mol: "Chem.Mol") -> torch.Tensor:
    idx = scaffold_atom_indices(mol)
    mask = torch.zeros((mol.GetNumAtoms(),), dtype=torch.bool)
    for i in idx:
        if 0 <= int(i) < int(mask.numel()):
            mask[int(i)] = True
    return mask
