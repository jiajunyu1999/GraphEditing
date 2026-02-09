from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .properties import build_property_fns, calc_properties


_PROCESS_PROPERTY_FNS: dict[str, Any] | None = None


def _init_process_worker(property_names: list[str]) -> None:
    global _PROCESS_PROPERTY_FNS
    _PROCESS_PROPERTY_FNS = build_property_fns(property_names)


def _compute_one_process(smiles: str) -> tuple[str, dict[str, float] | None, str | None]:
    assert _PROCESS_PROPERTY_FNS is not None
    return _compute_one(smiles, _PROCESS_PROPERTY_FNS)


def _compute_one(
    smiles: str, property_fns: dict[str, Any]
) -> tuple[str, dict[str, float] | None, str | None]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, None, "invalid_smiles"
    try:
        props = calc_properties(mol, property_fns)
    except Exception as e:
        return smiles, None, f"{type(e).__name__}: {e}"
    return smiles, props, None


@dataclass(frozen=True)
class PropertyResult:
    props_by_smiles: dict[str, dict[str, float]]
    errors_by_smiles: dict[str, str]


def compute_properties_multiprocess(
    smiles_list: list[str],
    *,
    property_names: list[str],
    num_workers: int,
    chunksize: int,
    executor: str,
) -> PropertyResult:
    unique = sorted({s for s in smiles_list if isinstance(s, str) and s})
    props_by_smiles: dict[str, dict[str, float]] = {}
    errors_by_smiles: dict[str, str] = {}

    property_fns = build_property_fns(property_names)

    def run_threaded() -> None:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for smiles, props, err in ex.map(
                lambda s: _compute_one(s, property_fns), unique, chunksize=chunksize
            ):
                if props is not None:
                    props_by_smiles[smiles] = props
                else:
                    errors_by_smiles[smiles] = err or "unknown_error"

    def run_process() -> None:
        # Note: some sandboxed environments block sysconf/sem limits used by ProcessPoolExecutor.
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_process_worker,
            initargs=(property_names,),
        ) as ex:
            for smiles, props, err in ex.map(_compute_one_process, unique, chunksize=chunksize):
                if props is not None:
                    props_by_smiles[smiles] = props
                else:
                    errors_by_smiles[smiles] = err or "unknown_error"

    if executor == "thread":
        run_threaded()
    elif executor == "process":
        run_process()
    elif executor == "auto":
        try:
            run_process()
        except (PermissionError, OSError):
            run_threaded()
    else:
        raise ValueError(f"Unknown executor: {executor}")

    return PropertyResult(props_by_smiles=props_by_smiles, errors_by_smiles=errors_by_smiles)


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_props{input_path.suffix}")


def main() -> None:
    # Suppress RDKit stderr warnings like "Can't kekulize mol ..."
    try:  # pragma: no cover
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Compute source/generated properties (and deltas) for a SMILES-edit dataset (multi-process)."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input", type=str, default=None, help="CSV path (defaults to config.output.csv_path)")
    parser.add_argument("--output", type=str, default=None, help="CSV path (default: <input>_props.csv)")
    parser.add_argument("--source_col", type=str, default="source_smiles")
    parser.add_argument("--generated_col", type=str, default="generated_smiles")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--executor", type=str, default="thread", choices=["auto", "thread", "process"])
    parser.add_argument(
        "--on_error",
        type=str,
        default="drop",
        choices=["nan", "drop"],
        help="If a SMILES fails, fill properties with NaN or drop rows.",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    property_names = list(cfg.raw["chemistry"]["property_names"])

    input_path = cfg.resolve_path(args.input) if args.input else cfg.resolve_path(cfg.raw["output"]["csv_path"])
    output_path = cfg.resolve_path(args.output) if args.output else _default_output_path(input_path)

    df = pd.read_csv(input_path, engine="c")
    for col in (args.source_col, args.generated_col):
        if col not in df.columns:
            raise KeyError(f"Missing column `{col}` in {input_path}. Columns: {list(df.columns)}")

    source_smiles = df[args.source_col].astype(str).tolist()
    generated_smiles = df[args.generated_col].astype(str).tolist()

    result = compute_properties_multiprocess(
        source_smiles + generated_smiles,
        property_names=property_names,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
        executor=args.executor,
    )

    for prop in property_names:
        df[f"{prop}_src"] = df[args.source_col].map(
            lambda s: result.props_by_smiles.get(str(s), {}).get(prop, float("nan"))
        )
        df[f"{prop}_gen"] = df[args.generated_col].map(
            lambda s: result.props_by_smiles.get(str(s), {}).get(prop, float("nan"))
        )
        df[f"{prop}_delta"] = df[f"{prop}_gen"] - df[f"{prop}_src"]

    if args.on_error == "drop":
        needed_cols = []
        for prop in property_names:
            needed_cols.extend([f"{prop}_src", f"{prop}_gen"])
        df = df.dropna(subset=needed_cols)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote -> {output_path}")
    if result.errors_by_smiles:
        print(f"Failed smiles: {len(result.errors_by_smiles)} (kept in output as NaN unless --on_error drop)")


if __name__ == "__main__":
    main()
