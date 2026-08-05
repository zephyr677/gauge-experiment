"""
analyze.py
==========
Legge i CSV in results/ e produce i tre grafici del progetto:

  fig1_sgd_control.png   - Fase 1: le curve SGD dei due arm devono coincidere
  fig2_adamw_gap.png     - Fase 2: media +/- std sui seed, per arm, con AdamW
  fig3_lr_sweep.png      - Fase 3: loss finale vs learning rate, per arm

Uso:  python src/analyze.py [--outdir results] [--tail 50]
"""

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r"(orig|rot|fisher)_(sgd|adamw)_lr([0-9.e+-]+)_seed(\d+)\.csv")
COL = {"orig": "#1f77b4", "rot": "#d62728", "fisher": "#2ca02c"}
LAB = {"orig": "originale", "rot": "ruotato", "fisher": "Fisher Q*"}


def load_all(outdir):
    rows = []
    for f in glob.glob(os.path.join(outdir, "*.csv")):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        arm, opt, lr, seed = m.group(1), m.group(2), float(m.group(3)), int(m.group(4))
        df = pd.read_csv(f)
        rows.append(dict(arm=arm, opt=opt, lr=lr, seed=seed, df=df))
    return rows


def smooth(y, k=9):
    if len(y) < k:
        return y
    return pd.Series(y).rolling(k, min_periods=1, center=True).mean().values


def fig1(runs, outdir):
    sgd = [r for r in runs if r["opt"] == "sgd"]
    if not sgd:
        return
    plt.figure(figsize=(8, 5))
    for r in sgd:
        plt.plot(r["df"].step, smooth(r["df"].loss.values),
                 color=COL[r["arm"]], alpha=0.9,
                 label=f"{LAB[r['arm']]} (lr={r['lr']:g})")
    # differenza assoluta fra le due curve, se presenti entrambe
    o = [r for r in sgd if r["arm"] == "orig"]
    t = [r for r in sgd if r["arm"] == "rot"]
    if o and t:
        n = min(len(o[0]["df"]), len(t[0]["df"]))
        d = np.abs(o[0]["df"].loss.values[:n] - t[0]["df"].loss.values[:n])
        print(f"[fase1] |delta loss| SGD: media={d.mean():.2e}  max={d.max():.2e}")
        print("        atteso: ~1e-4/1e-3 (rumore fp32). Se e' O(0.1) c'e' un bug.")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend()
    plt.title("Fase 1 - controllo SGD: i gemelli devono coincidere")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig1_sgd_control.png"), dpi=150)
    print("[fig] fig1_sgd_control.png")


def fig2(runs, outdir):
    ad = [r for r in runs if r["opt"] == "adamw"]
    lrs = sorted({r["lr"] for r in ad})
    # usa il LR con piu' seed disponibili
    best_lr = max(lrs, key=lambda l: len([r for r in ad if r["lr"] == l])) if lrs else None
    if best_lr is None:
        return
    plt.figure(figsize=(8, 5))
    for arm in ("orig", "rot", "fisher"):
        group = [r for r in ad if r["arm"] == arm and r["lr"] == best_lr]
        if not group:
            continue
        n = min(len(r["df"]) for r in group)
        Y = np.stack([r["df"].loss.values[:n] for r in group])
        mu, sd = Y.mean(0), Y.std(0)
        steps = group[0]["df"].step.values[:n]
        plt.plot(steps, smooth(mu), color=COL[arm],
                 label=f"{LAB[arm]} (n={len(group)} seed)")
        plt.fill_between(steps, smooth(mu - sd), smooth(mu + sd),
                         color=COL[arm], alpha=0.18)
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend()
    plt.title(f"Fase 2 - AdamW (lr={best_lr:g}): gap sistematico vs rumore dei seed")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig2_adamw_gap.png"), dpi=150)
    print("[fig] fig2_adamw_gap.png")


def fig3(runs, outdir, tail):
    ad = [r for r in runs if r["opt"] == "adamw"]
    if not ad:
        return
    plt.figure(figsize=(8, 5))
    for arm in ("orig", "rot", "fisher"):
        xs, ys = [], []
        for lr in sorted({r["lr"] for r in ad if r["arm"] == arm}):
            group = [r for r in ad if r["arm"] == arm and r["lr"] == lr]
            finals = [r["df"].loss.values[-tail:].mean() for r in group]
            xs.append(lr); ys.append(np.mean(finals))
        if xs:
            plt.plot(xs, ys, "o-", color=COL[arm], label=LAB[arm])
    plt.xscale("log"); plt.xlabel("learning rate"); plt.ylabel(f"loss media ultimi {tail} step")
    plt.legend(); plt.title("Fase 3 - sweep LR: stesso modello, coordinate diverse")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig3_lr_sweep.png"), dpi=150)
    print("[fig] fig3_lr_sweep.png")


def fig4(runs, outdir):
    ad = [r for r in runs if r["opt"] == "adamw" and "eval_loss" in r["df"].columns]
    ad = [r for r in ad if r["df"].eval_loss.notna().any()]
    if not ad:
        return
    plt.figure(figsize=(8, 5))
    for arm in ("orig", "rot", "fisher"):
        group = [r for r in ad if r["arm"] == arm]
        if not group:
            continue
        curves = []
        for r in group:
            e = r["df"].dropna(subset=["eval_loss"])
            curves.append((e.step.values, e.eval_loss.values))
        n = min(len(c[1]) for c in curves)
        Y = np.stack([c[1][:n] for c in curves])
        steps = curves[0][0][:n]
        mu, sd = Y.mean(0), Y.std(0)
        plt.plot(steps, mu, "o-", color=COL[arm], ms=3,
                 label=f"{LAB[arm]} (n={len(group)})")
        plt.fill_between(steps, mu - sd, mu + sd, color=COL[arm], alpha=0.18)
    plt.xlabel("step"); plt.ylabel("eval loss (held-out fisso)")
    plt.legend(); plt.title("Fase 2bis - eval loss su dati identici: il confronto pulito")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig4_eval.png"), dpi=150)
    print("[fig] fig4_eval.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results")
    p.add_argument("--tail", type=int, default=50)
    args = p.parse_args()
    runs = load_all(args.outdir)
    if not runs:
        print("Nessun CSV trovato in", args.outdir)
        return
    print(f"{len(runs)} run trovati.")
    fig1(runs, args.outdir)
    fig2(runs, args.outdir)
    fig3(runs, args.outdir, args.tail)
    fig4(runs, args.outdir)


if __name__ == "__main__":
    main()
