#!/usr/bin/env python3
"""
run_ablations.py
================
Ablation sweeps for the DP-MERF + voting extension.

Sweep order matters — run B first to find the best k_splits,
then use that in all subsequent sweeps.

Sweeps:
    B: k_splits         — how many independent pipelines to union (run first)
    A: vote_rounds      — iterative refinement rounds per split
    C: oversample_factor — extra candidates relative to n_per_split
    D: generator_fraction — generator vs random in the pool
    E: num_synth_factor — size of synthetic dataset relative to N
    F: k_splits x vote_rounds x num_synth_factor interaction

Usage
-----
    python run_ablations.py --sweep B1 # tests k_splits with voting round 1
    python run_ablations.py --sweep B5 # tests k_splits with voting round 2
    python run_ablations.py --sweep B10 # tests k_splits with voting round 10
    python run_ablations.py --sweep A --best_k_splits 2
    python run_ablations.py --sweep C --best_k_splits 2
    python run_ablations.py --sweep D --best_k_splits 2
    python run_ablations.py --sweep E --best_k_splits 2
    python run_ablations.py --sweep F
    python run_ablations.py --sweep all        # runs B,A,C,D,E,F in order
"""

import argparse
import os
import subprocess
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
#  Fixed settings — match your working command exactly               #
# ------------------------------------------------------------------ #

FIXED = dict(
    data="2d",
    noise_factor=5.0,
    synth_spec_string="norm_k5_n100000_row5_col5_noise0.2",
    epochs=50,
    gen_spec="200,500,500,200",
    rff_sigma="0.50",
    lr="3e-3",
    d_rff=30000,
    model_path="pt_models/epsilon_1.0/gen.pt",
    epsilon_vote=1.0,
    num_synth_factor=1.0,
    seed=42,
)


# ------------------------------------------------------------------ #
#  Sweep definitions                                                  #
# ------------------------------------------------------------------ #

# ---- Sweep B: k_splits — run this first ----
# vote_rounds=1, oversample_factor=0.5, all generator.
# Tests whether splitting into independent pipelines and unioning helps.
# k_splits=1 is the no-split baseline.
SWEEP_B1 = [
    dict(vote_rounds=1, k_splits=ks, oversample_factor=0.5, generator_fraction=1.0)
    for ks in [1, 5, 25, 50]
]

SWEEP_B5 = [
    dict(vote_rounds=5, k_splits=ks, oversample_factor=0.5, generator_fraction=1.0)
    for ks in [1, 5, 25, 50]
]

SWEEP_B10 = [
    dict(vote_rounds=10, k_splits=ks, oversample_factor=0.5, generator_fraction=1.0)
    for ks in [1, 5, 25, 50]
]

# ---- Sweep A: vote_rounds — iterative refinement per split ----
# Use best k_splits from B (default 2). k_splits=1 not re-tested.
SWEEP_A = [
    dict(vote_rounds=r, k_splits=2, oversample_factor=0.5, generator_fraction=1.0)
    for r in [1, 5, 10, 20]
]

# ---- Sweep C: oversample_factor — extra candidates relative to n_per_split ----
# Use best k_splits from B (default 2).
# oversample_factor=0.0 means no extra candidates (pool = current winners only).
# oversample_factor=1.0 means generate as many extra as we need to keep.
SWEEP_C = [
    dict(vote_rounds=1, k_splits=2, oversample_factor=of, generator_fraction=1.0)
    for of in [0.0, 0.25, 0.5, 0.75, 1.0]
]

# ---- Sweep D: generator_fraction — generator vs random in the pool ----
# Use best k_splits from B (default 2).
# generator_fraction=1.0 = all generator, =0.0 = all random (pure baseline).
SWEEP_D = [
    dict(vote_rounds=1, k_splits=2, oversample_factor=0.5, generator_fraction=gf)
    for gf in [1.0, 0.75, 0.5, 0.25, 0.0]
]

# ---- Sweep E: num_synth_factor — size of synthetic dataset relative to N ----
# Use best k_splits from B (default 2).
# Tests whether generating less data improves voting quality.
SWEEP_E = [
    dict(vote_rounds=1, k_splits=2, oversample_factor=0.5,
         generator_fraction=1.0, num_synth_factor=nsf)
    for nsf in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
]

# ---- Sweep F: k_splits x vote_rounds x num_synth_factor interaction ----
# All generator. Tests the joint effect of splits, rounds, and dataset size.
# k_splits=1 excluded since sweep B already showed it underperforms.
SWEEP_F = [
    dict(vote_rounds=vr, k_splits=ks, oversample_factor=0.5,
         generator_fraction=1.0, num_synth_factor=nsf)
    for vr, ks, nsf in [
        (1, 2, 1.0),    # 2 splits, 1 round, full dataset — baseline
        (1, 4, 1.0),    # 4 splits, 1 round, full dataset
        (1, 2, 0.5),    # 2 splits, 1 round, half dataset
        (1, 4, 0.5),    # 4 splits, 1 round, half dataset
        (2, 2, 1.0),    # 2 splits, 2 rounds, full dataset
        (2, 4, 1.0),    # 4 splits, 2 rounds, full dataset
        (2, 2, 0.5),    # 2 splits, 2 rounds, half dataset
        (2, 4, 0.5),    # 4 splits, 2 rounds, half dataset
    ]
]

SWEEPS = {"B1": SWEEP_B1, "B5": SWEEP_B5, "B10": SWEEP_B10,
          "A": SWEEP_A, "C": SWEEP_C,
          "D": SWEEP_D, "E": SWEEP_E, "F": SWEEP_F}
SWEEP_ORDER = ["B1", "B5", "B10", "A", "C", "D", "E", "F"]


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def config_label(cfg):
    if cfg.get("_label") == "baseline":
        return "no_vote_baseline"
    label = (
        f"ks{cfg['k_splits']}"
        f"_vr{cfg['vote_rounds']}"
        f"_of{cfg['oversample_factor']}"
        f"_gf{cfg['generator_fraction']}"
    )
    if cfg.get("num_synth_factor", 1.0) != 1.0:
        label += f"_nsf{cfg['num_synth_factor']}"
    return label


def run_one(cfg, fixed, base_log_dir, gen_script="gen_balanced.py"):
    """Launch gen_balanced.py as a subprocess for one config."""
    label   = config_label(cfg)
    log_dir = os.path.join(base_log_dir, label) + "/"

    cmd = [
        sys.executable, gen_script,
        "--data",               fixed["data"],
        "--log-dir",            log_dir,
        "--model_path",         fixed["model_path"],
        "--gen-spec",           fixed["gen_spec"],
        "--synth-spec-string",  fixed["synth_spec_string"],
        "--rff-sigma",          fixed["rff_sigma"],
        "--lr",                 fixed["lr"],
        "--d-rff",              str(fixed["d_rff"]),
        "--noise-factor",       str(fixed["noise_factor"]),
        "--epochs",             str(fixed["epochs"]),
        "--epsilon_vote",       str(fixed["epsilon_vote"]),
        "--num_synth_factor",   str(cfg.get("num_synth_factor", fixed["num_synth_factor"])),
        "--seed",               str(fixed["seed"]),
        "--vote_rounds",        str(cfg["vote_rounds"]),
        "--k_splits",           str(cfg["k_splits"]),
        "--oversample_factor",  str(cfg["oversample_factor"]),
        "--generator_fraction", str(cfg["generator_fraction"]),
    ]

    print("\n" + "="*70)
    print(f"[sweep] Running config: {label}")
    print("  " + " ".join(cmd))
    print("="*70)

    result = subprocess.run(cmd, capture_output=False)
    return log_dir, result.returncode


def read_eval_score(log_dir):
    """Try to read the scalar eval score written by test_results()."""
    path = os.path.join(log_dir, "eval_score")
    if os.path.isfile(path):
        with open(path) as f:
            try:
                return float(f.read().strip())
            except ValueError:
                pass
    path2 = os.path.join(log_dir, "final_score.json")
    if os.path.isfile(path2):
        with open(path2) as f:
            d = json.load(f)
            for v in d.values():
                if isinstance(v, (int, float)):
                    return float(v)
    return None


def plot_sweep_bar(sweep_name, configs, scores, base_log_dir):
    """Bar chart comparing all configs in a sweep by their eval score."""
    labels = [config_label(c) for c in configs]
    valid  = [(l, s) for l, s in zip(labels, scores) if s is not None]
    if not valid:
        print(f"[plot] No scores to plot for sweep {sweep_name}")
        return

    lbl, sc = zip(*valid)
    fig, ax = plt.subplots(figsize=(max(6, len(lbl) * 1.5), 5))
    bars = ax.bar(range(len(lbl)), sc, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(lbl)))
    ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Eval score (higher is better)")
    ax.set_title(f"Ablation {sweep_name}: per-config performance")

    for bar, val in zip(bars, sc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(base_log_dir, f"sweep_{sweep_name}_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] Saved -> {path}")


def _build_grid(sweep_name, configs, log_dirs, base_log_dir,
                plot_filename, out_filename, title_suffix):
    """Load one plot file per config and stitch into a comparison grid."""
    images, labels = [], []
    for cfg, d in zip(configs, log_dirs):
        path = os.path.join(d, plot_filename)
        if os.path.isfile(path):
            images.append(plt.imread(path))
            labels.append(config_label(cfg))

    if not images:
        print(f"[plot] No '{plot_filename}' images found for sweep {sweep_name}")
        return

    ncols = min(len(images), 4)
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, (img, lbl) in enumerate(zip(images, labels)):
        axes[i].imshow(img)
        axes[i].set_title(lbl, fontsize=8)
        axes[i].axis("off")
    for j in range(len(images), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Ablation {sweep_name} — {title_suffix}", fontsize=11, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(base_log_dir, out_filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved -> {out_path}")


def plot_combined_grid(sweep_name, configs, log_dirs, base_log_dir):
    """Produce both a subsampled and a centered comparison grid."""
    _build_grid(sweep_name, configs, log_dirs, base_log_dir,
                plot_filename="plot_gen_sub0.2.png",
                out_filename=f"sweep_{sweep_name}_grid_sub.png",
                title_suffix="subsampled (20%)")
    _build_grid(sweep_name, configs, log_dirs, base_log_dir,
                plot_filename="plot_gen_centered.png",
                out_filename=f"sweep_{sweep_name}_grid_centered.png",
                title_suffix="full dataset (centered)")


def print_summary(sweep_name, configs, scores):
    """Print a ranked table to stdout."""
    rows = [(config_label(c), s) for c, s in zip(configs, scores) if s is not None]
    rows.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'='*55}")
    print(f"Ablation {sweep_name} — ranking (best first)")
    print(f"{'='*55}")
    print(f"{'Config':<50} {'Score':>8}")
    print("-" * 60)
    for lbl, sc in rows:
        print(f"{lbl:<50} {sc:>8.4f}")
    if rows:
        print(f"\n Best config: {rows[0][0]}  (score = {rows[0][1]:.4f})")
    print("=" * 55)


def patch_sweeps(best_k_splits, best_vote_rounds, best_oversample, best_num_synth):
    """Patch later sweeps with best results from earlier ones."""
    if best_k_splits is not None:
        for cfg in SWEEP_A + SWEEP_C + SWEEP_D + SWEEP_E:
            cfg["k_splits"] = best_k_splits
    if best_vote_rounds is not None:
        for cfg in SWEEP_C + SWEEP_D + SWEEP_E:
            cfg["vote_rounds"] = best_vote_rounds
    if best_oversample is not None:
        for cfg in SWEEP_D + SWEEP_E:
            cfg["oversample_factor"] = best_oversample
    if best_num_synth is not None:
        for cfg in SWEEP_F:
            cfg["num_synth_factor"] = best_num_synth


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="B",
                    choices=["all"] + SWEEP_ORDER,
                    help="Which sweep to run. Start with B to find best k_splits.")
    ap.add_argument("--base_log_dir", default="logs/ablations")
    ap.add_argument("--gen_script",   default="gen_balanced.py")

    # CLI overrides for FIXED settings
    ap.add_argument("--model_path",        type=str,   default=None)
    ap.add_argument("--synth_spec_string", type=str,   default=None)
    ap.add_argument("--gen_spec",          type=str,   default=None)
    ap.add_argument("--rff_sigma",         type=str,   default=None)
    ap.add_argument("--lr",                type=str,   default=None)
    ap.add_argument("--d_rff",             type=int,   default=None)
    ap.add_argument("--epsilon_vote",      type=float, default=None)
    ap.add_argument("--epochs",            type=int,   default=None)
    ap.add_argument("--seed",              type=int,   default=None)

    # Best results from earlier sweeps — used to patch later sweep defaults
    ap.add_argument("--best_k_splits",    type=int,   default=None,
                    help="Best k_splits from sweep B. Patches A, C, D, E.")
    ap.add_argument("--best_vote_rounds", type=int,   default=None,
                    help="Best vote_rounds from sweep A. Patches C, D, E.")
    ap.add_argument("--best_oversample",  type=float, default=None,
                    help="Best oversample_factor from sweep C. Patches D, E.")
    ap.add_argument("--best_num_synth",   type=float, default=None,
                    help="Best num_synth_factor from sweep E. Patches F.")
    ap.add_argument("--baseline", action="store_true", default=False,
                    help="Run skip_vote baseline before the sweep for comparison")
    ap.add_argument("--baseline_only", action="store_true", default=False,
                    help="Run only the baseline, skip sweep configs")

    args = ap.parse_args()

    # Apply CLI overrides to FIXED
    fixed = dict(FIXED)
    for key in ["model_path", "synth_spec_string", "gen_spec", "rff_sigma",
                "lr", "d_rff", "epsilon_vote", "epochs", "seed"]:
        val = getattr(args, key)
        if val is not None:
            fixed[key] = val

    # Patch sweeps with best results from earlier sweeps
    patch_sweeps(args.best_k_splits, args.best_vote_rounds,
                 args.best_oversample, args.best_num_synth)
    
    if args.baseline_only:
        args.baseline = True

    sweeps_to_run = SWEEP_ORDER if args.sweep == "all" else [args.sweep]

    for sweep_name in sweeps_to_run:
        configs   = SWEEPS[sweep_name]
        sweep_dir = os.path.join(args.base_log_dir, f"sweep_{sweep_name}")
        os.makedirs(sweep_dir, exist_ok=True)

        log_dirs, scores = [], []

        if args.baseline:
            baseline_dir = os.path.join(sweep_dir, "baseline_no_vote") + "/"
            os.makedirs(baseline_dir, exist_ok=True)
            baseline_cmd = [
                sys.executable, args.gen_script,
                "--data",               fixed["data"],
                "--log-dir",            baseline_dir,
                "--model_path",         fixed["model_path"],
                "--gen-spec",           fixed["gen_spec"],
                "--synth-spec-string",  fixed["synth_spec_string"],
                "--rff-sigma",          fixed["rff_sigma"],
                "--lr",                 fixed["lr"],
                "--d-rff",              str(fixed["d_rff"]),
                "--noise-factor",       str(fixed["noise_factor"]),
                "--epochs",             str(fixed["epochs"]),
                "--seed",               str(fixed["seed"]),
                "--skip_vote",
            ]
            print("\n" + "="*70)
            print("[sweep] Running baseline (no voting)")
            print("  " + " ".join(baseline_cmd))
            print("="*70)
            subprocess.run(baseline_cmd, capture_output=False)
            baseline_score = read_eval_score(baseline_dir)
            print(f"  -> baseline score: {baseline_score}")

            baseline_cfg = dict(vote_rounds=0, k_splits=0, oversample_factor=0.0,
                                generator_fraction=1.0, _label="baseline")
            configs  = [baseline_cfg] + list(configs)
            log_dirs = [baseline_dir]
            scores   = [baseline_score]

        start_idx = 1 if args.baseline else 0
        if args.baseline_only:
            start_idx = len(configs)

        for cfg in configs[start_idx:]:
            log_dir, rc = run_one(cfg, fixed, sweep_dir, args.gen_script)
            log_dirs.append(log_dir)
            score = read_eval_score(log_dir)
            scores.append(score)
            print(f"  -> score: {score}")

        print_summary(sweep_name, configs, scores)
        plot_sweep_bar(sweep_name, configs, scores, sweep_dir)
        plot_combined_grid(sweep_name, configs, log_dirs, sweep_dir)

        summary = [{"config": config_label(c), "score": s}
                for c, s in zip(configs, scores)]
        with open(os.path.join(sweep_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[sweep] Summary -> {os.path.join(sweep_dir, 'summary.json')}")


if __name__ == "__main__":
    main()