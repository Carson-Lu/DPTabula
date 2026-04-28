#!/usr/bin/env python3
"""
run_ablations_tabular.py
========================
Ablation sweeps for DP-MERF + voting on real tabular datasets
using single_generator_priv_all.py.

Metrics:
    Binary datasets (adult, census, cervical, credit, epileptic, isolet):
        ROC-AUC and PRC averaged across classifiers
    Multiclass datasets (covtype, intrusion):
        F1 (weighted) averaged across classifiers

Sweep order:
    B: k_splits         — run first, find best k_splits
    A: vote_rounds      — iterative refinement
    C: oversample_factor
    D: generator_fraction
    E: num_synth_factor

Usage
-----
    python run_ablations_tabular.py --dataset adult --sweep B
    python run_ablations_tabular.py --dataset adult --sweep A --best_k_splits 2
    python run_ablations_tabular.py --dataset adult --sweep all
    python run_ablations_tabular.py --dataset covtype --sweep B
"""

import argparse
import os
import subprocess
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
#  Fixed settings per dataset                                         #
# ------------------------------------------------------------------ #

# Shared voting defaults — same across all datasets
VOTING_DEFAULTS = dict(
    epsilon_vote=0.5,
    epsilon_gen=0.5,
    num_synth_factor=1.0,
    oversample_factor=0.5,
    generator_fraction=1.0,
)

FIXED = dict(
    skip_vote_baseline=True,  # skip_vote for baseline
    epsilon_gen_baseline=1.0  # full epsilon for baseline
)

# Dataset-specific fixed settings (epochs, batch, undersample, num_features)
# These match the known good settings from the original paper
DATASET_FIXED = {
    "adult": dict(
        epochs="8000", batch=0.1, undersample=0.4,
        num_features="1000", repeat=3,
        classifiers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ),
    "census": dict(
        epochs="2000", batch=0.05, undersample=0.4,
        num_features="5000", repeat=3,
        classifiers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ),
    "cervical": dict(
        epochs="200", batch=0.1, undersample=1.0,
        num_features="2000", repeat=3,
        classifiers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ),
    "credit": dict(
        epochs="4000", batch=0.1, undersample=0.005,
        num_features="5000", repeat=3,
        classifiers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ),
    "epileptic": dict(
        epochs="6000", batch=0.01, undersample=1.0,
        num_features="80000", repeat=3,
        classifiers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ),
    "isolet": dict(
        epochs="4000", batch=0.1, undersample=1.0,
        num_features="500", repeat=3,
        classifiers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ),
    "covtype": dict(
        epochs="6000", batch=0.05, undersample=0.03,
        num_features="1000", repeat=3,
        classifiers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ),
    "intrusion": dict(
        epochs="10000", batch=0.03, undersample=0.3,
        num_features="2000", repeat=3,
        classifiers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ),
}

MULTICLASS_DATASETS = {"covtype", "intrusion"}
BINARY_DATASETS     = {"adult", "census", "cervical", "credit", "epileptic", "isolet"}


# ------------------------------------------------------------------ #
#  Sweep definitions                                                  #
# ------------------------------------------------------------------ #

# ---- Sweep B: k_splits — run this first ----
SWEEP_B1 = [
    dict(vote_rounds=1, k_splits=ks, oversample_factor=0.25, generator_fraction=1.0)
    for ks in [1, 5, 25, 50]
]

SWEEP_B5 = [
    dict(vote_rounds=5, k_splits=ks, oversample_factor=0.25, generator_fraction=1.0, repeat=3)
    for ks in [1, 5, 25, 50]
]

SWEEP_B10 = [
    dict(vote_rounds=10, k_splits=ks, oversample_factor=0.25, generator_fraction=1.0, repeat=3)
    for ks in [1, 5, 25, 50]
]


# ---- Sweep A: vote_rounds ----
# Use best k_splits from B (default 2)
SWEEP_A = [
    dict(vote_rounds=r, k_splits=25, oversample_factor=0.5, generator_fraction=1.0)
    for r in [1, 5, 10, 20, 30]
]

# ---- Sweep C: oversample_factor ----
SWEEP_C = [
    dict(vote_rounds=5, k_splits=25, oversample_factor=of, generator_fraction=1.0)
    for of in [0.25, 0.5, 0.75, 1.0]
]

# ---- Sweep D: generator_fraction ----
SWEEP_D = [
    dict(vote_rounds=5, k_splits=25, oversample_factor=0.25, generator_fraction=gf)
    for gf in [1.0, 0.75, 0.5, 0.25, 0.0]
]

# ---- Sweep E: num_synth_factor ----
SWEEP_E = [
    dict(vote_rounds=5, k_splits=25, oversample_factor=0.25,
         generator_fraction=1.0, num_synth_factor=nsf)
    for nsf in [0.25, 0.5, 0.75, 1.0]
]

# ---- Sweep F: epsilon distribution ----
SWEEP_F = [
    dict(vote_rounds=5, k_splits=25, oversample_factor=0.25, generator_fraction=1.0,
         epsilon_gen=eg, epsilon_vote=round(1.0 - eg, 2))
    for eg in [0.1, 0.25, 0.5, 0.75, 0.9]
]

SWEEP_BEST_ADULT = [
    dict(vote_rounds=5, k_splits=25, oversample_factor=0.25, epsilon_gen=0.9, epsilon_vote=0.1,
         generator_fraction=1.0, num_synth_factor=1.0, epochs=8000, num_features=10000, dataset="adult")
]

SWEEP_BEST_ALL = [
    dict(vote_rounds=5, k_splits=25, oversample_factor=0.25, epsilon_gen=0.9, epsilon_vote=0.1,
         generator_fraction=1.0, num_synth_factor=1.0, dataset=data)
    for data in ["adult", "census", "cervical", "credit", "epileptic", "isolet", "covtype", "intrusion"]
]

SWEEPS      = {"B1": SWEEP_B1, "B5": SWEEP_B5, "B10": SWEEP_B10, "A": SWEEP_A, "C": SWEEP_C, "D": SWEEP_D, "E": SWEEP_E, "F": SWEEP_F, "BEST_ADULT": SWEEP_BEST_ADULT, "BEST_ALL": SWEEP_BEST_ALL}
SWEEP_ORDER = ["B1", "B5", "B10", "A", "C", "D", "E", "F", "BEST_ADULT", "BEST_ALL"]


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
    if "epsilon_gen" in cfg:
        label += f"_eg{cfg['epsilon_gen']}_ev{cfg['epsilon_vote']}"
    return label


def run_one(cfg, dataset, voting_defaults, base_log_dir,
            gen_script="single_generator_priv_all.py"):
    """Launch single_generator_priv_all.py as a subprocess for one config."""
    label   = config_label(cfg)
    log_dir = os.path.join(base_log_dir, label)
    os.makedirs(log_dir, exist_ok=True)

    ds_fixed = DATASET_FIXED[dataset]
    epsilon_vote = cfg.get("epsilon_vote", voting_defaults["epsilon_vote"])
    epsilon_gen = cfg.get("epsilon_gen", voting_defaults["epsilon_gen"])
    num_synth_factor = cfg.get("num_synth_factor", voting_defaults["num_synth_factor"])

    cmd = [
        sys.executable, "-u", gen_script,
        "--dataset",            dataset,
        "--epochs",             ds_fixed["epochs"],
        "--batch",              str(ds_fixed["batch"]),
        "--undersample",        str(ds_fixed["undersample"]),
        "--num_features",       ds_fixed["num_features"],
        "--repeat",             str(ds_fixed["repeat"]),
        "--classifiers",        *[str(c) for c in ds_fixed["classifiers"]],
        "--epsilon_vote",       str(epsilon_vote),
        "--num_synth_factor",   str(num_synth_factor),
        "--vote_rounds",        str(cfg["vote_rounds"]),
        "--k_splits",           str(cfg["k_splits"]),
        "--oversample_factor",  str(cfg["oversample_factor"]),
        "--generator_fraction", str(cfg["generator_fraction"]),
        "--private",      "1",
        "--epsilon_gen",  str(epsilon_gen),
    ]

    print("\n" + "="*70)
    print(f"[sweep] Dataset: {dataset}  Config: {label}")
    print("  " + " ".join(cmd))
    print("="*70)

    # Redirect stdout to a log file and stream to console so progress is visible
    log_file = os.path.join(log_dir, "output.log")
    # Launch child unbuffered (we added -u) and stream stdout line-by-line
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    rc = None
    with open(log_file, "w", encoding="utf-8") as f:
        try:
            for line in process.stdout:
                f.write(line)
                f.flush()
                print(line, end="")
        except KeyboardInterrupt:
            process.kill()
            raise
        finally:
            process.wait()
            rc = process.returncode

    return log_dir, log_file, rc


def parse_score(log_file, dataset):
    """
    Parse the final average ROC+PRC (binary) or F1 (multiclass) from the log.
    Returns a dict with the relevant metrics.
    """
    if not os.path.isfile(log_file):
        return None

    with open(log_file) as f:
        lines = f.readlines()

    scores = {}
    for line in lines:
        line = line.strip()
        if "roc mean across methods" in line.lower():
            try:
                scores["roc"] = float(line.split("is")[-1].strip())
            except ValueError:
                pass
        if "prc mean across methods" in line.lower():
            try:
                scores["prc"] = float(line.split("is")[-1].strip())
            except ValueError:
                pass
        if "f1 mean across methods" in line.lower():
            try:
                scores["f1"] = float(line.split("is")[-1].strip())
            except ValueError:
                pass

    if not scores:
        return None

    # Primary metric for ranking
    if dataset in MULTICLASS_DATASETS:
        scores["primary"] = scores.get("f1", None)
    else:
        # use ROC + PRC combined as primary
        roc = scores.get("roc", 0)
        prc = scores.get("prc", 0)
        scores["primary"] = (roc + prc) / 2 if roc and prc else None

    return scores


def plot_sweep_bar(sweep_name, dataset, configs, scores, base_log_dir):
    """Bar chart of primary metric across configs."""
    labels  = [config_label(c) for c in configs]
    primary = [s["primary"] if s else None for s in scores]
    valid   = [(l, p) for l, p in zip(labels, primary) if p is not None]

    if not valid:
        print(f"[plot] No scores to plot for sweep {sweep_name}")
        return

    lbl, sc = zip(*valid)
    fig, ax  = plt.subplots(figsize=(max(6, len(lbl) * 1.5), 5))
    metric   = "F1" if dataset in MULTICLASS_DATASETS else "mean(ROC+PRC)"
    bars     = ax.bar(range(len(lbl)), sc, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(lbl)))
    ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"{metric} (higher is better)")
    ax.set_title(f"[{dataset}] Ablation {sweep_name}: per-config performance")

    for bar, val in zip(bars, sc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(base_log_dir, f"sweep_{sweep_name}_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] Saved -> {path}")


def plot_roc_prc_bars(sweep_name, dataset, configs, scores, base_log_dir):
    """For binary datasets, plot ROC and PRC as grouped bars."""
    if dataset in MULTICLASS_DATASETS:
        return

    labels = [config_label(c) for c in configs]
    rocs   = [s.get("roc") if s else None for s in scores]
    prcs   = [s.get("prc") if s else None for s in scores]
    valid  = [(l, r, p) for l, r, p in zip(labels, rocs, prcs)
              if r is not None and p is not None]

    if not valid:
        return

    lbl, roc_vals, prc_vals = zip(*valid)
    x   = np.arange(len(lbl))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(lbl) * 1.5), 5))
    ax.bar(x - w/2, roc_vals, w, label="ROC", color="steelblue", edgecolor="white")
    ax.bar(x + w/2, prc_vals, w, label="PRC", color="coral",     edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title(f"[{dataset}] Ablation {sweep_name}: ROC and PRC")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(base_log_dir, f"sweep_{sweep_name}_roc_prc.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] Saved -> {path}")


def print_summary(sweep_name, dataset, configs, scores):
    """Print a ranked table to stdout."""
    rows = [(config_label(c), s) for c, s in zip(configs, scores) if s is not None]
    rows.sort(key=lambda x: x[1]["primary"] if x[1]["primary"] else -1, reverse=True)

    print(f"\n{'='*60}")
    print(f"Ablation {sweep_name} — [{dataset}] ranking (best first)")
    print(f"{'='*60}")

    if dataset in MULTICLASS_DATASETS:
        print(f"{'Config':<45} {'F1':>8}")
        print("-" * 55)
        for lbl, s in rows:
            f1 = s.get("f1", "N/A")
            print(f"{lbl:<45} {f1:>8.4f}" if isinstance(f1, float) else f"{lbl:<45} {'N/A':>8}")
    else:
        print(f"{'Config':<45} {'ROC':>8} {'PRC':>8} {'Mean':>8}")
        print("-" * 72)
        for lbl, s in rows:
            roc  = s.get("roc", "N/A")
            prc  = s.get("prc", "N/A")
            mean = s.get("primary", "N/A")
            print(f"{lbl:<45} {roc:>8.4f} {prc:>8.4f} {mean:>8.4f}"
                  if all(isinstance(v, float) for v in [roc, prc, mean])
                  else f"{lbl:<45} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

    if rows and rows[0][1]["primary"]:
        print(f"\n Best: {rows[0][0]}  (score = {rows[0][1]['primary']:.4f})")
    print("=" * 60)


def patch_sweeps(best_k_splits, best_vote_rounds, best_oversample):
    if best_k_splits is not None:
        for cfg in SWEEP_A + SWEEP_C + SWEEP_D + SWEEP_E:
            cfg["k_splits"] = best_k_splits
    if best_vote_rounds is not None:
        for cfg in SWEEP_C + SWEEP_D + SWEEP_E:
            cfg["vote_rounds"] = best_vote_rounds
    if best_oversample is not None:
        for cfg in SWEEP_D + SWEEP_E:
            cfg["oversample_factor"] = best_oversample


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adult",
                    choices=list(DATASET_FIXED.keys()),
                    help="Dataset to run ablations on")
    ap.add_argument("--sweep", default="B",
                    choices=["all"] + SWEEP_ORDER)
    ap.add_argument("--base_log_dir", default="logs/ablations_tabular")
    ap.add_argument("--gen_script",   default="single_generator_priv_all.py")

    # Patch later sweeps with best from earlier
    ap.add_argument("--best_k_splits",    type=int,   default=None)
    ap.add_argument("--best_vote_rounds", type=int,   default=None)
    ap.add_argument("--best_oversample",  type=float, default=None)

    # Override voting defaults
    ap.add_argument("--epsilon_vote",    type=float, default=None)
    ap.add_argument("--num_synth_factor",type=float, default=None)
    ap.add_argument("--epsilon_gen_baseline", type=float, default=None)
    ap.add_argument("--epsilon_gen",     type=float, default=None)
    
    ap.add_argument("--baseline", action="store_true", default=False,
                help="Run skip_vote baseline before the sweep for comparison")
    ap.add_argument("--baseline_only", action="store_true", default=False,
                help="Run only the original no-vote baseline (skip sweep configs)")

    args = ap.parse_args()

    if args.baseline_only:
        args.baseline = True

    voting_defaults = dict(VOTING_DEFAULTS)
    if args.epsilon_vote is not None:
        voting_defaults["epsilon_vote"] = args.epsilon_vote
    if args.num_synth_factor is not None:
        voting_defaults["num_synth_factor"] = args.num_synth_factor
    if args.epsilon_gen_baseline is not None:
        FIXED["epsilon_gen_baseline"] = args.epsilon_gen_baseline
    if args.epsilon_gen is not None:
        voting_defaults["epsilon_gen"] = args.epsilon_gen

    patch_sweeps(args.best_k_splits, args.best_vote_rounds, args.best_oversample)

    sweeps_to_run = SWEEP_ORDER if args.sweep == "all" else [args.sweep]

    for sweep_name in sweeps_to_run:
        configs   = SWEEPS[sweep_name]
        sweep_dir = os.path.join(args.base_log_dir, args.dataset, f"sweep_{sweep_name}")
        os.makedirs(sweep_dir, exist_ok=True)

        log_dirs, all_scores = [], []   # initialise first
        
        ds_fixed = DATASET_FIXED[args.dataset]  # add this line
        # ---- baseline run ----
        if args.baseline:
            baseline_cmd = [
                sys.executable, args.gen_script,
                "--dataset",      args.dataset,
                "--epochs",       ds_fixed["epochs"],
                "--batch",        str(ds_fixed["batch"]),
                "--undersample",  str(ds_fixed["undersample"]),
                "--num_features", ds_fixed["num_features"],
                "--repeat",       str(ds_fixed["repeat"]),
                "--classifiers",  *[str(c) for c in ds_fixed["classifiers"]],
                "--skip_vote",
                "--private",      "1",                              # add this
                "--epsilon_gen",  str(FIXED["epsilon_gen_baseline"]), # add this
            ]
            baseline_dir = os.path.join(sweep_dir, "baseline_no_vote")
            os.makedirs(baseline_dir, exist_ok=True)
            baseline_log = os.path.join(baseline_dir, "output.log")
            with open(baseline_log, "w") as f:
                subprocess.run(baseline_cmd, stdout=f, stderr=subprocess.STDOUT)
            with open(baseline_log) as f:
                print(f.read())
            baseline_score = parse_score(baseline_log, args.dataset)
            print(f"  Baseline (no vote) score: {baseline_score}")

            baseline_cfg = dict(vote_rounds=1, k_splits=1, oversample_factor=0.5,
                            generator_fraction=1.0, _label="baseline")
            configs    = [baseline_cfg] + list(configs)
            log_dirs   = [baseline_dir]
            all_scores = [baseline_score]

        start_idx = 1 if args.baseline else 0
        if args.baseline_only:
            start_idx = len(configs)

        for cfg in configs[start_idx:]:
            log_dir, log_file, rc = run_one(
                cfg, args.dataset, voting_defaults, sweep_dir, args.gen_script)
            log_dirs.append(log_dir)
            scores = parse_score(log_file, args.dataset)
            all_scores.append(scores)
            if scores:
                print(f"  -> primary score: {scores['primary']:.4f}")
            else:
                print(f"  -> score: None")

        print_summary(sweep_name, args.dataset, configs, all_scores)
        plot_sweep_bar(sweep_name, args.dataset, configs, all_scores, sweep_dir)
        plot_roc_prc_bars(sweep_name, args.dataset, configs, all_scores, sweep_dir)

        summary = [{"config": config_label(c), "scores": s}
                   for c, s in zip(configs, all_scores)]
        with open(os.path.join(sweep_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[sweep] Summary -> {os.path.join(sweep_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
