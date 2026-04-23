#!/usr/bin/env python3
"""
run_ablations_v2.py
===================
Ablation sweeps for the split-vote-union DP-MERF extension.
Works with the updated gen_balanced.py args:
    --n_splits, --vote_rounds, --oversample_factor,
    --generator_fraction, --random_fraction, --epsilon_vote

Four sweeps, run one at a time:
    Sweep A: vote_rounds      — iterative refinement rounds per split
    Sweep B: n_splits         — independent pipelines to union
    Sweep C: oversample_factor — how many candidates relative to needed
    Sweep D: generator_fraction — generator vs random in the pool

Usage
-----
    python run_ablations_v2.py --sweep A
    python run_ablations_v2.py --sweep B
    python run_ablations_v2.py --sweep C
    python run_ablations_v2.py --sweep D
    python run_ablations_v2.py --sweep all
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

# Key constraint: candidates per split should not greatly exceed private
# points per class. With num_synth_factor=1:
#   candidates = n_per_class_split * oversample_factor
#              = (n_per_class / n_splits) * oversample_factor
#   private    = n_private_per_class ~ n_per_class
# So: oversample_factor <= n_splits keeps candidates <= private points.

# ---- Sweep A: vote_rounds — iterative refinement per split ----
# Fixed: n_splits=1, oversample_factor=1.25, all generator
SWEEP_A = [
    dict(vote_rounds=r, n_splits=1, oversample_factor=1.25, generator_fraction=1.0, random_fraction=0.0)
    for r in [1, 2, 3, 5]
]

# ---- Sweep B: n_splits — independent pipelines to union ----
# Fixed: vote_rounds=1, oversample_factor=1.25, all generator
# As n_splits increases, candidates per split decrease relative to private points.
SWEEP_B = [
    dict(vote_rounds=1, n_splits=ns, oversample_factor=1.25, generator_fraction=1.0, random_fraction=0.0)
    for ns in [1, 2, 4, 8]
]

# ---- Sweep C: oversample_factor — candidates relative to needed ----
# Fixed: n_splits=2, vote_rounds=1, all generator.
# With n_splits=2, oversample_factor<=2 keeps candidates<=private points.
SWEEP_C = [
    dict(vote_rounds=1, n_splits=2, oversample_factor=of, generator_fraction=1.0, random_fraction=0.0)
    for of in [1.0, 1.25, 1.5, 1.75, 2.0]
]

# ---- Sweep D: generator_fraction — generator vs random in the pool ----
# Fixed: n_splits=2, vote_rounds=1, oversample_factor=1.25.
# generator_fraction=1.0 = all generator, =0.0 = all random (pure baseline).
SWEEP_D = [
    dict(vote_rounds=1, n_splits=2, oversample_factor=1.25, generator_fraction=gf, random_fraction=round(1.0-gf, 2))
    for gf in [1.0, 0.75, 0.5, 0.25, 0.0]
]

# ---- Sweep E: num_synth_factor — size of synthetic dataset relative to N ----
# Fixed: n_splits=1, vote_rounds=1, oversample_factor=1.25, all generator.
# Tests whether generating less data improves quality (better bins-to-data ratio)
# or if generating more data is worth the quality tradeoff.
SWEEP_E = [
    dict(vote_rounds=1, n_splits=1, oversample_factor=1.25, generator_fraction=1.0,
         random_fraction=0.0, num_synth_factor=nsf)
    for nsf in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
]

# ---- Sweep F: vote_rounds x n_splits x num_synth_factor (generator only) ----
# Tests the interaction between how many splits/rounds we use and how much
# data we want to generate. All generator, no random.
# The hypothesis: smaller num_synth_factor + more splits may outperform
# larger num_synth_factor + fewer splits since the bins-to-data ratio stays healthy.
SWEEP_F = [
    dict(vote_rounds=vr, n_splits=ns, oversample_factor=1.25, generator_fraction=1.0,
         random_fraction=0.0, num_synth_factor=nsf)
    for vr, ns, nsf in [
        (1, 1, 1.0),    # baseline: single pipeline, full dataset
        (1, 2, 1.0),    # 2 splits, full dataset
        (1, 4, 1.0),    # 4 splits, full dataset
        (1, 1, 0.5),    # single pipeline, half dataset
        (1, 2, 0.5),    # 2 splits, half dataset — each split generates N/4
        (1, 4, 0.5),    # 4 splits, half dataset — each split generates N/8
        (2, 2, 0.5),    # 2 splits + 2 rounds, half dataset
        (2, 2, 1.0),    # 2 splits + 2 rounds, full dataset
    ]
]

SWEEPS = {"A": SWEEP_A, "B": SWEEP_B, "C": SWEEP_C, "D": SWEEP_D, "E": SWEEP_E, "F": SWEEP_F}


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def config_label(cfg):
    """Short human-readable label for a config dict."""
    label = (
        f"vr{cfg['vote_rounds']}"
        f"_ns{cfg['n_splits']}"
        f"_of{cfg['oversample_factor']}"
        f"_gf{cfg['generator_fraction']}"
    )
    # only append num_synth_factor if it differs from the default (1.0)
    if cfg.get("num_synth_factor", 1.0) != 1.0:
        label += f"_nsf{cfg['num_synth_factor']}"
    return label


def run_one(cfg, fixed, base_log_dir, gen_script="gen_balanced.py"):
    """Launch gen_balanced.py as a subprocess for one config."""
    label = config_label(cfg)
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
        "--n_splits",           str(cfg["n_splits"]),
        "--oversample_factor",  str(cfg["oversample_factor"]),
        "--generator_fraction", str(cfg["generator_fraction"]),
        "--random_fraction",    str(cfg["random_fraction"]),
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
    fig, ax = plt.subplots(figsize=(max(6, len(lbl) * 1.2), 5))
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
    print(f"[plot] Saved → {path}")


def _build_grid(sweep_name, configs, log_dirs, base_log_dir, plot_filename, out_filename, title_suffix):
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
    print(f"[plot] Saved → {out_path}")


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
    print(f"\n{'='*50}")
    print(f"Ablation {sweep_name} — ranking (best first)")
    print(f"{'='*50}")
    print(f"{'Config':<50} {'Score':>8}")
    print("-" * 60)
    for lbl, sc in rows:
        print(f"{lbl:<50} {sc:>8.4f}")
    if rows:
        print(f"\n✓ Best config: {rows[0][0]}  (score = {rows[0][1]:.4f})")
    print("=" * 50)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="all", choices=["all", "A", "B", "C", "D", "E", "F"])
    ap.add_argument("--base_log_dir", default="logs/ablations")
    ap.add_argument("--gen_script", default="gen_balanced.py")

    # Allow overriding any FIXED setting from the command line
    ap.add_argument("--model_path",         type=str,   default=None)
    ap.add_argument("--synth_spec_string",  type=str,   default=None)
    ap.add_argument("--gen_spec",           type=str,   default=None)
    ap.add_argument("--rff_sigma",          type=str,   default=None)
    ap.add_argument("--lr",                 type=str,   default=None)
    ap.add_argument("--d_rff",              type=int,   default=None)
    ap.add_argument("--epsilon_vote",       type=float, default=None)
    ap.add_argument("--epochs",             type=int,   default=None)
    ap.add_argument("--seed",               type=int,   default=None)

    # Patch later sweeps with best result from earlier sweeps
    ap.add_argument("--best_vote_rounds",    type=int,   default=None,
                    help="Patch sweeps B/C/D with best vote_rounds from sweep A")
    ap.add_argument("--best_n_splits",       type=int,   default=None,
                    help="Patch sweeps C/D with best n_splits from sweep B")
    ap.add_argument("--best_oversample",     type=float, default=None,
                    help="Patch sweep D with best oversample_factor from sweep C")
    ap.add_argument("--best_num_synth",      type=float, default=None,
                    help="Override num_synth_factor in sweep F with best value from sweep E")

    args = ap.parse_args()

    # Build fixed dict, applying any CLI overrides
    fixed = dict(FIXED)
    for key in ["model_path", "synth_spec_string", "gen_spec", "rff_sigma",
                "lr", "d_rff", "epsilon_vote", "epochs", "seed"]:
        val = getattr(args, key)
        if val is not None:
            fixed[key] = val

    # Patch later sweeps with best results from earlier ones
    if args.best_vote_rounds is not None:
        for cfg in SWEEP_B + SWEEP_C + SWEEP_D:
            cfg["vote_rounds"] = args.best_vote_rounds
    if args.best_n_splits is not None:
        for cfg in SWEEP_C + SWEEP_D:
            cfg["n_splits"] = args.best_n_splits
    if args.best_oversample is not None:
        for cfg in SWEEP_D:
            cfg["oversample_factor"] = args.best_oversample
    if args.best_num_synth is not None:
        for cfg in SWEEP_F:
            cfg["num_synth_factor"] = args.best_num_synth

    sweeps_to_run = ["A", "B", "C", "D", "E", "F"] if args.sweep == "all" else [args.sweep]

    for sweep_name in sweeps_to_run:
        configs   = SWEEPS[sweep_name]
        sweep_dir = os.path.join(args.base_log_dir, f"sweep_{sweep_name}")
        os.makedirs(sweep_dir, exist_ok=True)

        log_dirs, scores = [], []

        for cfg in configs:
            log_dir, rc = run_one(cfg, fixed, sweep_dir, args.gen_script)
            log_dirs.append(log_dir)
            score = read_eval_score(log_dir)
            scores.append(score)
            print(f"  → score: {score}")

        print_summary(sweep_name, configs, scores)
        plot_sweep_bar(sweep_name, configs, scores, sweep_dir)
        plot_combined_grid(sweep_name, configs, log_dirs, sweep_dir)

        summary = [{"config": config_label(c), "score": s}
                   for c, s in zip(configs, scores)]
        with open(os.path.join(sweep_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[sweep] Summary → {os.path.join(sweep_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
