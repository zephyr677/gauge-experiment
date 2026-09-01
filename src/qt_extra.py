"""
qt_extra.py
===========
Chiude il confronto fra predittori del danno aggiungendo i punti FUORI dalla
famiglia Q(t) — quelli dove le metriche possono disaccordare:

  nativa (gap 0) | Q(t) t=0.25..1 | Haar (gap 1.01) | Q* Fisher (gap ~0)

e introducendo la metrica giusta al posto della massa L1 off-diagonale:

  kappa_prec = numero di condizionamento della Fisher PRECONDIZIONATA
               kappa( D^{-1/2} F' D^{-1/2} ),  D = diag(F')

che e' la definizione operativa di "quanto e' buono un precondizionatore
diagonale in questa base". Riportata in log10.

Stadi (dalla radice del progetto):
  python src/qt_extra.py --stage metrics   # CPU, ~1 min (serve F.pt e qt_A.pt)
  python src/qt_extra.py --stage cv        # L4, ~15 min (sonda su Haar e Q*)
  python src/qt_extra.py --stage scatter   # CPU: figura finale + Spearman

Prerequisiti: F.pt e qt_A.pt (da qt_sweep --stage fisher/build),
models/rotated (fase 0), models/fisher (fisher_basis + rotate_model --q-file;
se assente il punto Q* viene saltato con un avviso).
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch

T_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
LR = 3e-4

# punto -> (cartella modello, csv della loss a 3e-4)
POINTS = {
    "native": ("models/original", "results_sweep/orig_adamw_lr0.0003_seed0.csv"),
    "haar":   ("models/rotated",  "results_sweep/rot_adamw_lr0.0003_seed0.csv"),
    "qstar":  ("models/fisher",   "results_sweep/fisher_adamw_lr0.0003_seed0.csv"),
}


def get_Q(name):
    """Q della base rispetto alla nativa (float64)."""
    F = torch.load("F.pt")
    d = F.shape[0]
    if name == "native":
        return torch.eye(d, dtype=torch.float64)
    if name == "haar":
        return torch.load("models/rotated/Q.pt").double()
    if name == "qstar":
        return torch.load("Qstar.pt").double()
    # Q(t)
    A = torch.load("qt_A.pt")
    return torch.linalg.matrix_exp(float(name) * A)


def both_metrics(Q, F):
    Fp = Q.T @ F @ Q
    off = ((Fp - torch.diag(torch.diagonal(Fp))).abs().sum() / Fp.abs().sum()).item()
    dg = torch.diagonal(Fp).clamp_min(1e-12)
    W = Fp / torch.sqrt(dg.unsqueeze(0) * dg.unsqueeze(1))   # D^-1/2 F' D^-1/2
    lam = 1e-6 * torch.trace(W).item() / W.shape[0]
    ev = torch.linalg.eigvalsh(0.5 * (W + W.T) + lam * torch.eye(W.shape[0], dtype=torch.float64))
    kappa = (ev.max() / ev.clamp_min(1e-12).min()).item()
    return off, torch.log10(torch.tensor(kappa)).item()


def stage_metrics():
    F = torch.load("F.pt")
    out = {}
    names = [f"{t:g}" for t in T_VALUES] + ["haar"] + (["qstar"] if os.path.exists("Qstar.pt") else [])
    if "qstar" not in names:
        print("AVVISO: Qstar.pt assente -> punto Q* saltato "
              "(rigenera con: python src/fisher_basis.py)")
    for n in names:
        off, lk = both_metrics(get_Q(n), F)
        out[n] = {"offdiag": off, "log10_kappa_prec": lk}
        print(f"[{n:>6}] offdiag={off:.4f}   log10 kappa_prec={lk:.3f}")
    json.dump(out, open("qt_extra_metrics.json", "w"))


def stage_cv():
    from qt_sweep import stage_orth  # riusa la sonda, ma su modelli specifici
    # sonda manuale identica a qt_sweep.stage_orth, sui due modelli extra
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train import build_dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("models/original")
    blocks = build_dataset(tok)[:-32]
    perm = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(0))
    PROBE_LAYERS, BS, WARMUP = [0, 7, 15, 23, 29], 8, 50
    out = json.load(open("qt_extra_cv.json")) if os.path.exists("qt_extra_cv.json") else {}
    def has_weights(p):
        return os.path.exists(os.path.join(p, "model.safetensors")) or \
               os.path.exists(os.path.join(p, "pytorch_model.bin"))
    todo = [("haar", "models/rotated")] + ([("qstar", "models/fisher")]
                                           if has_weights("models/fisher") else [])
    if len(todo) == 1:
        print("AVVISO: models/fisher assente -> CV di Q* saltato")
    for name, path in todo:
        if name in out:
            continue
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device)
        model.gradient_checkpointing_enable(); model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                eps=1e-8, weight_decay=0.0)
        sel = [m for i, l in enumerate(model.model.layers) if i in PROBE_LAYERS
               for m in (l.self_attn.q_proj, l.self_attn.k_proj, l.self_attn.v_proj,
                         l.self_attn.o_proj, l.mlp.gate_proj, l.mlp.up_proj,
                         l.mlp.down_proj)]
        cvs = []
        for step in range(120):
            for grp in opt.param_groups:
                grp["lr"] = LR * min(1.0, (step + 1) / WARMUP)
            idx = perm[(step * BS) % len(blocks): (step * BS) % len(blocks) + BS]
            batch = blocks[idx].to(device)
            loss = model(input_ids=batch, labels=batch).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step >= 80 and (step - 80) % 10 == 0:
                with torch.no_grad():
                    for mod in sel:
                        st = opt.state[mod.weight]
                        Aup = st["exp_avg"] / (st["exp_avg_sq"].sqrt() + 1e-8)
                        s = torch.linalg.svdvals(Aup.float())
                        cvs.append((s.std(unbiased=False) / s.mean()).item())
        out[name] = sum(cvs) / len(cvs)
        print(f"[{name}] CV update = {out[name]:.4f}")
        json.dump(out, open("qt_extra_cv.json", "w"))
        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()
    json.dump(out, open("qt_extra_cv.json", "w"))


def stage_scatter(tail=50):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    met = json.load(open("qt_extra_metrics.json"))
    cv_qt = {f"{float(k):g}": v for k, v in json.load(open("qt_cv.json")).items()}
    cv_ex = json.load(open("qt_extra_cv.json"))

    rows = []
    base = pd.read_csv(POINTS["native"][1]).loss.values[-tail:].mean()
    for t in T_VALUES:
        n = f"{t:g}"
        f = f"results_qt/t{n}/rot_adamw_lr{LR:g}_seed0.csv"
        loss = pd.read_csv(f).loss.values[-tail:].mean()
        rows.append(dict(name=f"t={n}", gap=loss - base, cv=cv_qt[n], **met[n]))
    for name in ["haar", "qstar"]:
        if name not in met or name not in cv_ex or not os.path.exists(POINTS[name][1]):
            continue
        loss = pd.read_csv(POINTS[name][1]).loss.values[-tail:].mean()
        rows.append(dict(name=name, gap=loss - base, cv=cv_ex[name], **met[name]))
    d = pd.DataFrame(rows)
    d.to_csv("results_qt/summary_full.csv", index=False)
    print(d.to_string(index=False))

    print("\nSpearman sul set COMPLETO (famiglia Q(t) + fuori-famiglia):")
    for col, lab in [("offdiag", "offdiag L1 (vecchia)"),
                     ("log10_kappa_prec", "log10 kappa_prec (nuova)"),
                     ("cv", "CV update (Maes/Zhang)")]:
        rho = d[["gap", col]].corr(method="spearman").iloc[0, 1]
        print(f"  gap ~ {lab:26s}: {rho:+.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    for i, (col, lab) in enumerate([("offdiag", "offdiag L1"),
                                    ("log10_kappa_prec", "log10 kappa precondizionata"),
                                    ("cv", "CV update (Maes/Zhang)")]):
        ax[i].scatter(d[col], d.gap, c="#333")
        for _, r in d.iterrows():
            ax[i].annotate(r["name"], (r[col], r.gap), fontsize=8,
                           xytext=(4, 4), textcoords="offset points")
        rho = d[["gap", col]].corr(method="spearman").iloc[0, 1]
        ax[i].set_xlabel(lab); ax[i].set_title(f"Spearman rho = {rho:+.2f}")
        ax[i].set_ylabel("gap di loss vs nativa" if i == 0 else "")
    plt.tight_layout()
    plt.savefig("results_qt/fig_qt_full.png", dpi=150)
    print("[fig] results_qt/fig_qt_full.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["metrics", "cv", "scatter"])
    args = p.parse_args()
    {"metrics": stage_metrics, "cv": stage_cv, "scatter": stage_scatter}[args.stage]()
