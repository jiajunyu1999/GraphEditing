# Molecule Editing

This repository contains the code demo for **UniEdit:A Graph-based MoE Alternative to Sequence Generation for Molecular Editing**, accepted as a **KDD 2026 Oral** paper. It demonstrates one-shot molecule editing using a pretrained checkpoint on a small test set.

Please download the pretrained checkpoint before running the demo.

## What's Included

* `infer_oneshot.py`: Entry point for one-shot molecule editing inference.
* `infer_config.yaml`: Demo inference configuration, including model settings, task definitions, and input/output paths.
* `model.pth`: Pretrained checkpoint (download required; see below).
* `data/test_chatdrug.csv`: Sample input molecules stored in the `mol` column.
* `tasks.yaml`: Task definitions and corresponding property optimization directions.
* `evaluate.py`: Optional script for evaluating the generated molecules.

## Quickstart

From the repository root:

### 1. Download the Checkpoint

Download `model.pth` from:

https://drive.google.com/file/d/1t4GDDtgKR69I9Y6BNEWqG9AOKQ3BQdgs/view?usp=drive_link

Place it at:

```text
Molecular_Editing/model.pth
```

i.e., in the same directory as this README.

### 2. Run Inference

```bash
python infer_oneshot.py --infer_config infer_config.yaml
```

The script generates one CSV file per task under `output_` (e.g., `output_/101_task.csv`). Each output file contains a `pred_smiles_list_json` column with the generated molecules.

## Evaluation (Optional)

```bash
cd demo
python evaluate.py \
  --test_data data/test_chatdrug.csv \
  --result_prefix output_ \
  --config_path tasks.yaml \
  --mol-col mol
```

## Dependencies

* Python 3.8+
* RDKit (recommended via `conda-forge`)
* PyTorch
* `pandas`
* `pyyaml`

See `../requirements.txt` for additional dependency details.

## Notes

* The demo uses `cuda:0` by default.
* To run on CPU, set `device: cpu` in `infer_config.yaml` or pass `--device cpu`.
