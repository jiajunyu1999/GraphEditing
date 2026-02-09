# Molecule Editing

This folder is a demo for running one-shot molecule editing with a pretrained
checkpoint on a small test set. Download the checkpoint before running.

## What's included
- `infer_oneshot.py`: entry point for inference.
- `infer_config.yaml`: demo inference config (model, task defs, input/output paths).
- `model.pth`: pretrained checkpoint (download required, see below).
- `data/test_chatdrug.csv`: sample input molecules (`mol` column).
- `tasks.yaml`: task definitions and property trends.
- `evaluate.py`: optional evaluation script for the generated outputs.

## Quickstart
From the repo root:

### Download checkpoint
Download `model.pth` from:
https://drive.google.com/file/d/1t4GDDtgKR69I9Y6BNEWqG9AOKQ3BQdgs/view?usp=drive_link

Place the file at `Molecular_Editing/model.pth` (same folder as this README).

```bash
python infer_oneshot.py --infer_config infer_config.yaml
```

This writes one CSV per task to `output_` (e.g., `output_/101_task.csv`), with a
`pred_smiles_list_json` column containing the generated molecules.

## Evaluate (optional)
```bash
cd demo
python evaluate.py \
  --test_data data/test_chatdrug.csv \
  --result_prefix output_ \
  --config_path tasks.yaml \
  --mol-col mol
```

## Dependencies
- Python 3.8+
- RDKit (recommended via conda-forge)
- PyTorch
- `pandas`, `pyyaml` (see `../requirements.txt`)

## Notes
- The demo uses `cuda:0` by default. To run on CPU, update `infer_config.yaml`
  (`device: cpu`) or pass `--device cpu`.

