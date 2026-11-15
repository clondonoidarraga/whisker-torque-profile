# Whisker-base signals: position, shape and surface relief (WHISKiT Physics)

Anonymized code, notebooks and results accompanying the submission
"Contact Type Determines What Whisker-Base Signals Reveal: Position, Shape and Surface Relief in a Simulated Whisking Array".

Moments and forces at the base of the 27 right-side whiskers of the rat are generated with
WHISKiT Physics (https://github.com/SeNSE-lab/whiskitphysics) and used to train models that
estimate the position, the shape family and the surface relief of an object.

## Contents

- `dataset/`: simulation generators (`run_dataset*.py`), packing (`build_npz*.py`), training
  (`train_A.py` peg localization, `train_B.py` shape and position, `train_S.py` profile reconstruction),
  robustness (`robustness_A.py`) and ablation (`ablation_A.py`).
- `patches/whiskit_patches.diff`: changes to WHISKiT Physics (object position, yaw and scale;
  head trajectory; command-line options).
- `notebooks/`: Google Colab notebooks that build WHISKiT and generate each dataset (CPU only).
- `results/`: result files (JSON) and logs of every experiment in the paper.
- `setup_whiskit.sh`: clones WHISKiT, applies the patches and builds it.

## Correspondence with the paper

| Paper | Files | Results |
|---|---|---|
| Dataset P, Table II | `run_dataset.py`, `train_A.py` | `results/results_A_s2`, `_s5`, `_s10` |
| Table III (architecture and loss) | `ablation_A.py` | `results/ablation_A`, `results/ablation_A_large` |
| Table IV (robustness) | `robustness_A.py` | `results/robust_A` |
| Datasets D1, D2, D3, Table V | `run_dataset_B.py`, `_B2.py`, `_B3.py`, `train_B.py` | `results/results_B`, `_B2`, `_B3` |
| Dataset S, Tables VI and VII | `run_dataset_S.py`, `train_S.py` | `results/results_S_v2`, `results_S_aug`, `results_S_sin_extremos` |

Comments in the code and notebooks are in Spanish.

## Reproduction

1. `./setup_whiskit.sh` (Linux or Colab) builds `whiskitphysics/code/build/whiskit`.
2. Generate data, e.g. `cd dataset && python3 run_dataset.py ...`, then `python3 build_npz.py --dir <folder> --out dataset_A.npz`.
3. Train: `python3 train_A.py --data dataset_A.npz --folds 5 --seeds 3 --epochs 200 --stride 2`,
   `python3 train_B.py --data dataset_B2.npz`, or
   `python3 train_S.py --data dataset_S.npz dataset_S2.npz --seeds 3 --clip 5`.

The `.npz` datasets (hundreds of MB) are not versioned; they are regenerated with the notebooks.

## Protocol

- One sample is one full simulation (time x 27 whiskers x 6 channels: Mx, My, Mz, Fx, Fy, Fz).
- Split by simulation, five-fold cross-validation (stratified for shape), 20% internal validation for early stopping.
- Normalization fitted on training data only; several seeds; errors in mm.
- Baselines: training mean, ridge, random forest on the contact map, and a pose-only control that detects design shortcuts.

WHISKiT Physics belongs to its authors and keeps its own license; this repository contains patches only.
