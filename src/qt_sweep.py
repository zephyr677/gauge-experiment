"""
qt_sweep.py
===========
Esperimento Q(t): dose-risposta CONTINUA della rotazione, con funzione
preservata a ogni punto, e confronto testa-a-testa fra due predittori del danno:

  - la NOSTRA metrica:  massa off-diagonale della Fisher di blocco in base
    ruotata,  offdiag(Q_t^T F Q_t)
  - la LORO metrica (Maes/Zhang 2024, sez. 4.2): coefficiente di variazione
    dei valori singolari dell'update effettivo di Adam  A = m/(sqrt(v)+eps),
    per layer (CV basso = update quasi-ortogonale = prestazioni buone)

Costruzione: A antisimmetrica casuale riscalata a norma spettrale pi,
Q(t) = expm(t*A). Ortogonale per ogni t => gemello funzionalmente identico
per ogni t. t=0 e' l'originale; t=1 e' una rotazione "forte" (angoli fino a pi).

Stadi (lanciare dalla RADICE del progetto):
  python src/qt_sweep.py --stage build     # CPU, ~15 min: gemelli + check logit
  python src/qt_sweep.py --stage fisher    # GPU, ~10 min: F.pt + offdiag(t)
  python src/qt_sweep.py --stage train     # GPU, ~2 h: un run AdamW per t
  python src/qt_sweep.py --stage orth      # GPU, ~35 min: CV update per t
  python src/qt_sweep.py --stage analyze   # CPU: figura + correlazioni

Costo totale stimato: ~2 crediti L4.
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
import argparse
import json
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rotate_model import rotate                       # la rotazione validata
from train import build_dataset                       # stessa pipeline dati

T_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
ORIG = "models/original"                              # gia' untied + norm assorbite
LR = 3e-4                                             # dove il danno e' massimo
STEPS, WARMUP, BS = 400, 50, 8
PROBE_LAYERS = [0, 7, 15, 23, 29]                     # layer per il CV update


def tdir(t):
    return ORIG if t == 0.0 else f"models/qt{t:g}"


# ----------------------------------------------------------------------
def make_A(d, seed):
    """Antisimmetrica casuale, riscalata a norma spettrale pi (float64)."""
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(d, d, generator=g, dtype=torch.float64)
    A = (G - G.T) / 2.0
    A = A * (torch.pi / torch.linalg.matrix_norm(A, 2))
    return A


@torch.no_grad()
def stage_build(seed):
    tok = AutoTokenizer.from_pretrained(ORIG)
    base = AutoModelForCausalLM.from_pretrained(ORIG, dtype=torch.float32).eval()
    d = base.config.hidden_size
    A = make_A(d, seed)
    torch.save(A, "qt_A.pt")

    prompts = ["The capital of France is", "def fibonacci(n):",
               "Once upon a time", "1, 1, 2, 3, 5, 8,"]
    ref = [base(tok(p, return_tensors="pt").input_ids).logits for p in prompts]

    for t in T_VALUES:
        if t == 0.0:
            continue
        Q = torch.linalg.matrix_exp(t * A)
        err = (Q.T @ Q - torch.eye(d, dtype=torch.float64)).abs().max().item()
        m = AutoModelForCausalLM.from_pretrained(ORIG, dtype=torch.float32).eval()
        m = rotate(m, Q)
        # check logit inline
        worst = 0.0
        for p, r in zip(prompts, ref):
            out = m(tok(p, return_tensors="pt").input_ids).logits
            worst = max(worst, ((out - r).abs().max() / r.abs().max()).item())
        m.save_pretrained(tdir(t)); tok.save_pretrained(tdir(t))
        torch.save(Q, f"{tdir(t)}/Q.pt")
        status = "PASS" if worst < 1e-3 else "FAIL !!!"
        print(f"[t={t:g}] ortho={err:.1e}  logit rel err={worst:.2e}  {status}")
        assert worst < 1e-3, "invarianza rotta: non proseguire"
    print("build OK: tutti i gemelli sono funzionalmente identici all'originale")


# ----------------------------------------------------------------------
def residual_mats(model):
    for layer in model.model.layers:
        yield "r", layer.self_attn.q_proj
        yield "r", layer.self_attn.k_proj
        yield "r", layer.self_attn.v_proj
        yield "w", layer.self_attn.o_proj
        yield "r", layer.mlp.gate_proj
        yield "r", layer.mlp.up_proj
        yield "w", layer.mlp.down_proj
    yield "r", model.lm_head
    yield "r", model.model.embed_tokens


def stage_fisher(batches=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ORIG)
    if not os.path.exists("F.pt"):
        model = AutoModelForCausalLM.from_pretrained(ORIG, dtype=torch.float32).to(device)
        model.gradient_checkpointing_enable(); model.train()
        d = model.config.hidden_size
        blocks = build_dataset(tok)[:-32]
        perm = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(0))
        F = torch.zeros(d, d, dtype=torch.float64, device=device)
        for b in range(batches):
            idx = perm[(b * BS) % len(blocks): (b * BS) % len(blocks) + BS]
            batch = blocks[idx].to(device)
            loss = model(input_ids=batch, labels=batch).loss
            model.zero_grad(set_to_none=True); loss.backward()
            with torch.no_grad():
                for kind, mod in residual_mats(model):
                    G = mod.weight.grad.double()
                    F += G.T @ G if kind == "r" else G @ G.T
            if b % 16 == 0:
                print(f"[fisher] batch {b}/{batches}")
        F = (0.5 * (F + F.T)).cpu()
        torch.save(F, "F.pt")
    F = torch.load("F.pt")

    def offdiag(M):
        return ((M - torch.diag(torch.diagonal(M))).abs().sum() / M.abs().sum()).item()

    A = torch.load("qt_A.pt")
    out = {}
    for t in T_VALUES:
        Q = torch.linalg.matrix_exp(t * A) if t > 0 else torch.eye(F.shape[0], dtype=torch.float64)
        out[t] = offdiag(Q.T @ F @ Q)
        print(f"[t={t:g}] massa off-diagonale Fisher = {out[t]:.4f}")
    json.dump(out, open("qt_offdiag.json", "w"))


# ----------------------------------------------------------------------
def stage_train():
    for t in T_VALUES:
        outdir = f"results_qt/t{t:g}"
        if os.path.exists(f"{outdir}/rot_adamw_lr{LR:g}_seed0.csv"):
            print(f"[t={t:g}] gia' fatto, salto"); continue
        subprocess.run([sys.executable, "src/train.py",
                        "--model-dir", tdir(t), "--arm", "rot",
                        "--optimizer", "adamw", "--lr", str(LR), "--seed", "0",
                        "--steps", str(STEPS), "--batch-size", str(BS),
                        "--warmup", str(WARMUP), "--outdir", outdir], check=True)


# ----------------------------------------------------------------------
def stage_orth(probe_steps=120, measure_from=80, every=10):
    """CV dei valori singolari di A = m/(sqrt(v)+eps) — metrica Maes/Zhang."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ORIG)
    blocks = build_dataset(tok)[:-32]
    perm = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(0))
    out = {}
    for t in T_VALUES:
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(tdir(t), dtype=torch.float32).to(device)
        model.gradient_checkpointing_enable(); model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                eps=1e-8, weight_decay=0.0)
        sel = [m for i, l in enumerate(model.model.layers) if i in PROBE_LAYERS
               for m in (l.self_attn.q_proj, l.self_attn.k_proj, l.self_attn.v_proj,
                         l.self_attn.o_proj, l.mlp.gate_proj, l.mlp.up_proj,
                         l.mlp.down_proj)]
        cvs = []
        for step in range(probe_steps):
            for grp in opt.param_groups:
                grp["lr"] = LR * min(1.0, (step + 1) / WARMUP)
            idx = perm[(step * BS) % len(blocks): (step * BS) % len(blocks) + BS]
            batch = blocks[idx].to(device)
            loss = model(input_ids=batch, labels=batch).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step >= measure_from and (step - measure_from) % every == 0:
                with torch.no_grad():
                    for mod in sel:
                        st = opt.state[mod.weight]
                        Aup = st["exp_avg"] / (st["exp_avg_sq"].sqrt() + 1e-8)
                        s = torch.linalg.svdvals(Aup.float())
                        cvs.append((s.std(unbiased=False) / s.mean()).item())
        out[t] = sum(cvs) / len(cvs)
        print(f"[t={t:g}] CV update (media su {len(cvs)} misure) = {out[t]:.4f}")
        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()
    json.dump(out, open("qt_cv.json", "w"))


# ----------------------------------------------------------------------
def stage_analyze(tail=50):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    offd = {float(k): v for k, v in json.load(open("qt_offdiag.json")).items()}
    cv = {float(k): v for k, v in json.load(open("qt_cv.json")).items()}
    rows = []
    for t in T_VALUES:
        f = f"results_qt/t{t:g}/rot_adamw_lr{LR:g}_seed0.csv"
        df = pd.read_csv(f)
        rows.append(dict(t=t, loss=df.loss.values[-tail:].mean(),
                         offdiag=offd[t], cv=cv[t]))
    d = pd.DataFrame(rows)
    d["gap"] = d.loss - d.loss.iloc[0]
    d.to_csv("results_qt/summary.csv", index=False)
    print(d.to_string(index=False))

    rho_f = d[["gap", "offdiag"]].corr(method="spearman").iloc[0, 1]
    rho_c = d[["gap", "cv"]].corr(method="spearman").iloc[0, 1]
    print(f"\nSpearman gap~offdiag(Fisher): {rho_f:+.3f}")
    print(f"Spearman gap~CV(update):      {rho_c:+.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].plot(d.t, d.loss, "o-", color="#d62728")
    ax[0].set_xlabel("t (intensita' della rotazione)"); ax[0].set_ylabel(f"loss media ultimi {tail} step")
    ax[0].set_title("Dose-risposta: danno vs t")
    ax[1].plot(d.t, d.offdiag, "s-", color="#2ca02c", label="off-diag Fisher (nostra)")
    ax[1].plot(d.t, d.cv, "^-", color="#9467bd", label="CV update (Maes/Zhang)")
    ax[1].set_xlabel("t"); ax[1].legend(); ax[1].set_title("Le due metriche vs t")
    ax[2].scatter(d.offdiag, d.gap, color="#2ca02c",
                  label=f"Fisher  (rho={rho_f:+.2f})")
    ax2b = ax[2].twiny()
    ax2b.scatter(d.cv, d.gap, color="#9467bd", marker="^",
                 label=f"CV  (rho={rho_c:+.2f})")
    ax[2].set_xlabel("off-diag Fisher", color="#2ca02c")
    ax2b.set_xlabel("CV update", color="#9467bd")
    ax[2].set_ylabel("gap di loss vs t=0")
    ax[2].set_title("Predittore vs danno")
    ax[2].legend(loc="upper left"); ax2b.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results_qt/fig_qt.png", dpi=150)
    print("[fig] results_qt/fig_qt.png")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["build", "fisher", "train", "orth", "analyze"])
    p.add_argument("--seed", type=int, default=42, help="seed della A antisimmetrica")
    args = p.parse_args()
    {"build": lambda: stage_build(args.seed),
     "fisher": stage_fisher,
     "train": stage_train,
     "orth": stage_orth,
     "analyze": stage_analyze}[args.stage]()
