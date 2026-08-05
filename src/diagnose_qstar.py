"""
diagnose_qstar.py
=================
Verdetto sul pareggio Fisher Q* vs base nativa. Zero modifiche ad altri file.

Fa tre cose:
  1. analizza Qstar.pt: e' una permutazione con segni (ipotesi A) o densa (B)?
  2. curtosi delle norme di colonna per orig / rot / fisher
  3. sanity check: confronta i primi step dei CSV fisher vs orig allo stesso LR

Uso (dalla radice del progetto):
  cd ~/gauge-experiment
  python src/diagnose_qstar.py
"""

import glob
import os
import sys

# funziona da qualunque cartella: si sposta sulla radice del progetto
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from transformers import AutoModelForCausalLM

LINE = "-" * 60

# ---------------------------------------------------------------
# 1) Struttura di Q*
# ---------------------------------------------------------------
print(LINE)
print("1) STRUTTURA DI Q*  (perm+segni = ipotesi A, densa = ipotesi B)")
print(LINE)
if not os.path.exists("Qstar.pt"):
    print("Qstar.pt non trovato nella radice del progetto!")
    sys.exit(1)

Q = torch.load("Qstar.pt").abs()
d = Q.shape[0]
peak = Q.max(dim=0).values          # permutazione pura -> tutti 1.0
pr = 1.0 / (Q ** 4).sum(dim=0)      # participation ratio: perm=1, densa ~ d/3

print(f"picco medio per colonna:        {peak.mean():.3f}   (perm=1.00, densa~0.15)")
print(f"colonne con picco > 0.9:        {(peak > 0.9).float().mean() * 100:.1f}%")
print(f"colonne con picco > 0.5:        {(peak > 0.5).float().mean() * 100:.1f}%")
print(f"participation ratio medio:      {pr.mean():.1f}  su d={d}  (perm=1, densa~{d // 3})")

if peak.mean() > 0.8:
    verdict_q = "A"
    print(">> Q* e' quasi una PERMUTAZIONE CON SEGNI -> ipotesi A")
elif peak.mean() < 0.35:
    verdict_q = "B"
    print(">> Q* e' una rotazione DENSA -> ipotesi B")
else:
    verdict_q = "mista"
    print(">> Q* e' intermedia: parte degli assi preservati, parte mescolati")

# ---------------------------------------------------------------
# 2) Curtosi per i tre modelli
# ---------------------------------------------------------------
print()
print(LINE)
print("2) CURTOSI NORME DI COLONNA  (outlier: alta=preservati, ~3=spalmati)")
print(LINE)

def kurt(x):
    x = x.float().flatten()
    x = (x - x.mean()) / (x.std() + 1e-9)
    return (x ** 4).mean().item()

for tag, path in [("orig  ", "models/original"),
                  ("rot   ", "models/rotated"),
                  ("fisher", "models/fisher")]:
    if not os.path.isdir(path):
        print(f"{tag}: cartella {path} non trovata, salto")
        continue
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    ks = []
    for layer in m.model.layers:
        for proj in (layer.self_attn.q_proj, layer.mlp.up_proj):
            ks.append(kurt(proj.weight.norm(dim=0)))
    t = torch.tensor(ks)
    print(f"{tag}: curtosi media = {t.mean():6.2f}   max = {t.max():7.2f}")
    del m

# ---------------------------------------------------------------
# 3) Sanity check sui CSV: fisher NON deve essere copia identica di orig
# ---------------------------------------------------------------
print()
print(LINE)
print("3) SANITY CHECK CSV  (identici a tutte le cifre = artefatto!)")
print(LINE)

found = False
for f_fis in sorted(glob.glob("results_sweep/fisher_adamw_lr*_seed0.csv")):
    f_org = f_fis.replace("fisher_", "orig_")
    if not os.path.exists(f_org):
        continue
    found = True
    a = pd.read_csv(f_fis).loss.values[:60]
    b = pd.read_csv(f_org).loss.values[:60]
    n = min(len(a), len(b))
    diff = abs(a[:n] - b[:n])
    lr = f_fis.split("_lr")[1].split("_")[0]
    status = "!!! IDENTICI: ARTEFATTO, run da rifare" if diff.max() == 0.0 else "ok, run distinti"
    print(f"lr={lr}: |delta loss| primi {n} step  media={diff.mean():.2e}  max={diff.max():.2e}  -> {status}")

if not found:
    print("nessuna coppia fisher/orig trovata in results_sweep/")

# ---------------------------------------------------------------
# Verdetto
# ---------------------------------------------------------------
print()
print(LINE)
print("VERDETTO")
print(LINE)
if verdict_q == "A":
    print("""Ipotesi A: la base nativa E' (a meno di permutazione e segni)
l'autobase della Fisher del task. Adam e' invariante per permutazioni
e flip di segno, quindi il pareggio e' strutturale: non si puo' battere
la base nativa diagonalizzando la Fisher, perche' la nativa la
diagonalizza gia'. Il pretraining con Adam produce la base che rende
Adam un buon approssimatore del natural gradient.""")
elif verdict_q == "B":
    print("""Ipotesi B: Q* e' una rotazione densa eppure pareggia la base nativa.
Qualunque base che diagonalizza la Fisher del task vale quanto la
nativa: la nativa non ha proprieta' speciali oltre alla
diagonalizzazione. (Verifica che la curtosi di fisher sia ~3.)""")
else:
    print("""Caso misto: la Fisher e' quasi-diagonale su un sottoinsieme di
dimensioni (probabilmente le outlier, che Q* preserva) e mescolata sul
resto (dove pero' l'energia e' bassa e ad Adam non importa). Guarda la
percentuale di colonne con picco>0.9 e la curtosi di fisher: se la
curtosi resta alta, la sostanza e' comunque l'ipotesi A sulle
dimensioni che contano.""")
