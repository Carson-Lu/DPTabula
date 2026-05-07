# DP-MERF + Voting: Differentially Private Synthetic Tabular Data Generation

This project extends [DP-MERF](https://arxiv.org/abs/2002.11603) with a nearest-neighbour histogram voting post-processing step to improve the quality of differentially private synthetic tabular data. The core idea is to use private data to vote on a pool of generated candidates, retaining those closest to the real data distribution, while splitting generation into $k$ independent pipelines to improve vote signal quality.

## Overview

DP-MERF trains a generator by privately releasing a kernel mean embedding of the training data, after which synthetic data can be drawn at no additional privacy cost. We augment this with a voting refinement step that:

- Generates candidate synthetic samples using the DP-MERF generator
- Uses private data to vote for the nearest candidates via a noisy histogram
- Splits generation into `k` independent pipelines, each operating on a smaller candidate pool, and unions their outputs
- Preserves formal `(ε, δ)`-DP guarantees through sequential composition and parallel composition across classes

Results show consistent improvements over DP-MERF on binary classification datasets (Adult, Census, Credit), with up to 10% improvement in downstream classification performance.

---

## Repository Structure

```
DP-MERF/
├── code_tab/                          # Tabular data experiments
│   ├── single_generator_priv_all.py   # Main training and generation script for tabular datasets
│   ├── run_ablations_tabular.py       # Automated ablation sweeps for tabular experiments
│   └── utils/
│       └── histogram_voting.py        # Core voting utilities (shared across all scripts)
│
└── code_balanced/                     # 2D Gaussian mixture experiments
    ├── gen_balanced.py                # Training and generation script for Gaussian data
    └── run_ablations.py               # Automated ablation sweeps for Gaussian experiments
```

---

## Key Files

### `utils/histogram_voting.py`
The core shared library. Contains all voting logic used by both the tabular and Gaussian experiments:
- `run_voting_pipeline()` — outer wrapper managing splits, noise calibration, and unioning results
- `run_voting()` — core voting loop for one split across all classes
- `generate_samples()` — mixes generator and random candidates according to `generator_fraction`
- `generate_random()` — uniform random candidate generation over the feature space
- `vote()` — nearest-neighbour histogram voting with Gaussian noise
- `find_required_noise_multiplier()` — calibrates noise `σ` to achieve target `ε_vote`
- `build_columns_info()` — builds column metadata for mixed numerical/categorical tabular data
- `features_to_samples()` / `samples_to_features()` — conversion utilities between numpy arrays and the list format used by the voting functions

### `code_tab/single_generator_priv_all.py`
Main script for tabular data experiments. Handles:
- Data loading and preprocessing for Adult, Census, Credit, Covtype, and Intrusion datasets
- DP-MERF generator training with configurable privacy budget `ε_gen`
- Model saving and loading to avoid retraining across ablation runs
- Voting post-processing via `run_voting_pipeline()`
- Evaluation across 12 downstream classifiers (ROC, PRC, F1)

Key arguments:
```
--dataset          Dataset to use (adult, census, credit, covtype, intrusion)
--private          Enable DP training (1 = yes)
--epsilon_gen      Privacy budget for generator training
--skip_vote        Skip voting, use raw generator output
--vote_rounds      Number of iterative voting rounds (T)
--k_splits         Number of independent pipelines to union (K)
--oversample_factor  Extra candidates per round relative to n_per_split (ρ)
--generator_fraction  Fraction of candidates from generator vs random (α)
--epsilon_vote     Privacy budget for voting
--num_synth_factor  Size of synthetic dataset relative to training set (η)
--model_dir        Directory to save/load trained models
```

### `code_tab/run_ablations_tabular.py`
Automated sweep runner for tabular experiments. Runs multiple configurations sequentially, captures logs, parses ROC/PRC/F1 scores, and generates bar charts and summary JSON files.

```bash
# Run k_splits sweep on Adult (recommended first sweep)
python run_ablations_tabular.py --dataset adult --sweep B1 --baseline

# Run vote_rounds sweep using best k_splits
python run_ablations_tabular.py --dataset adult --sweep A --best_k_splits 25

# Run best config across all datasets
python run_ablations_tabular.py --sweep BEST_ALL
```

### `code_balanced/gen_balanced.py`
Training and generation script for 2D Gaussian mixture experiments. Uses `FCCondGen`/`ConvCondGen` as the generator and integrates the same voting pipeline.

### `code_balanced/run_ablations.py`
Automated sweep runner for Gaussian experiments. Produces visual comparison grids and NLL score summaries across configurations.

---

## Reproducing Results

### Tabular Datasets

**1. Train baseline (no voting):**
```bash
python single_generator_priv_all.py --dataset adult --private 1 \
    --epsilon_gen 1.0 --skip_vote --epochs 8000 --num_features 1000 \
    --batch 0.1 --undersample 0.4 --repeat 3
```

**2. Train with voting:**
```bash
python single_generator_priv_all.py --dataset adult --private 1 \
    --epsilon_gen 0.75 --epsilon_vote 0.25 \
    --vote_rounds 5 --k_splits 25 --oversample_factor 0.25 \
    --generator_fraction 1.0 --epochs 8000 --num_features 1000 \
    --batch 0.1 --undersample 0.4 --repeat 3
```

**3. Run real data baseline:**
```bash
python single_generator_priv_all.py --dataset adult --data_type real \
    --undersample 0.4 --epochs 200 --num_features 2000 --repeat 3
```

### Gaussian Data

**1. Train baseline:**
```bash
python gen_balanced.py --data 2d --skip_vote --noise-factor 5.0 \
    --epochs 50 --model_path pt_models/epsilon_1.0/gen.pt
```

**2. Train with voting:**
```bash
python gen_balanced.py --data 2d --noise-factor 5.0 --epochs 50 \
    --model_path pt_models/epsilon_1.0/gen.pt \
    --vote_rounds 5 --k_splits 50 --oversample_factor 0.5 \
    --generator_fraction 1.0 --epsilon_vote 0.9
```

---

## Privacy Accounting

The total privacy budget is split between the generator and voting step:

```
ε = ε_gen + ε_vote
δ = δ_gen + δ_vote = 1e-5 + 1/(n log n)
```

The voting step uses `T × K` histogram releases. Since each training point belongs to exactly one class, parallel composition applies across classes and the total cost is `T × K` rather than `T × K × |C|`. Noise `σ` is calibrated using `find_required_noise_multiplier()` to ensure the total voting cost does not exceed `ε_vote`.

---

## Best Hyperparameters (Adult Dataset)

| Parameter | Value |
|---|---|
| `k_splits` (K) | 25 |
| `vote_rounds` (T) | 5 |
| `oversample_factor` (ρ) | 0.25 |
| `generator_fraction` (α) | 1.0 |
| `num_synth_factor` (η) | 1.0 |
| `ε_gen` | 0.75 |
| `ε_vote` | 0.25 |

---

## Dependencies

```
torch
numpy
pandas
scikit-learn
autodp
sdgym
xgboost
scipy
matplotlib
seaborn
```
