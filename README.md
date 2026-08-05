# Gauge O(d) del residual stream: Adam se ne accorge?

**Domanda:** un LLM riscritto in coordinate ruotate (funzione IDENTICA) si addestra
in modo diverso con AdamW? SGD e' equivariante per costruzione; AdamW no, perche'
il suo precondizionatore e' diagonale nella base delle coordinate.

**Design:** due "gemelli" — `orig` (SmolLM2-135M con RMSNorm assorbite) e `rot`
(lo stesso, ruotato con Q ortogonale globale). Stessi dati, stesso ordine,
stesso tutto. L'unica variabile e' il sistema di coordinate.

## Struttura

```
src/rotate_model.py   costruisce i gemelli (assorbe norm, ruota, salva)
src/verify.py         Fase 0: i logit devono coincidere (PASS/FAIL)
src/train.py          un singolo run deterministico -> CSV
src/analyze.py        i 3 grafici del progetto
run_phases.sh         orchestrazione
```

## Setup su Lightning (CPU Studio, gratis)

```bash
pip install -r requirements.txt
bash run_phases.sh 0        # rotazione (CPU, ~5 min) + verifica
```

Se `verify.py` stampa **PASS** (errore < 1e-3, atteso ~1e-5) passa alla GPU.
Se **FAIL**: c'e' un bug, non accendere la L4.

## Fasi su L4

```bash
bash run_phases.sh 1        # controllo SGD    ~30 min  (~0.4 crediti)
bash run_phases.sh 2        # AdamW 3 seed     ~2.5 h   (~1.8 crediti)
bash run_phases.sh 3        # sweep LR         ~3 h     (~2.1 crediti)
```

Ogni fase produce/aggiorna i grafici in `results/`:

- `fig1_sgd_control.png` — le due curve DEVONO sovrapporsi (valida la pipeline)
- `fig2_adamw_gap.png`   — gap fra arm vs banda dei seed: il risultato
- `fig3_lr_sweep.png`    — loss vs LR per arm: se le curve non coincidono,
  il LR ottimale dipende dalle coordinate

## Interpretazione

| Esito Fase 2 | Lettura |
|---|---|
| gap dentro la banda dei seed | AdamW e' di fatto gauge-robusto (negativo pulito) |
| gap sistematico piccolo | il gauge e' un iperparametro minore |
| gap grande / LR ottimale diverso | il gauge e' un iperparametro non riconosciuto |

## Note tecniche

- Tutto in **fp32** (il controllo SGD in bf16 non funzionerebbe: troppa deriva numerica).
- Il `--seed` di train.py controlla SOLO l'ordine dei dati, identico per i due arm.
- Grad clipping globale e weight decay decoupled sono gauge-equivarianti: ok tenerli.
- Le RMSNorm sono assorbite in ENTRAMBI gli arm: i gemelli differiscono solo per Q.
- `verify.py` va rilanciato ogni volta che tocchi `rotate_model.py`.
