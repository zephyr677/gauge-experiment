#!/usr/bin/env bash
# =====================================================================
# Orchestrazione delle fasi. Lancia UNA fase alla volta:
#   bash run_phases.sh 0     # rotazione + verifica  (CPU ok, GPU per verify)
#   bash run_phases.sh 1     # controllo SGD         (~30 min L4)
#   bash run_phases.sh 2     # AdamW, 3 seed x 2 arm (~2.5 h L4)
#   bash run_phases.sh 3     # sweep LR              (~3 h L4)
#   bash run_phases.sh plot  # grafici               (CPU)
# =====================================================================
set -e

MODEL="HuggingFaceTB/SmolLM2-135M"
STEPS=400
BS=8

case "$1" in
0)
  python src/rotate_model.py --model $MODEL \
      --out-original models/original --out-rotated models/rotated --seed 1234
  python src/verify.py --original models/original --rotated models/rotated
  ;;
1)
  # SGD, un lr moderato, stesso seed dati: le curve devono coincidere
  for ARM in orig rot; do
    DIR=models/original; [ $ARM = rot ] && DIR=models/rotated
    python src/train.py --model-dir $DIR --arm $ARM \
        --optimizer sgd --lr 1e-3 --seed 0 --steps $STEPS --batch-size $BS
  done
  python src/analyze.py
  ;;
2)
  # AdamW: 3 seed (= 3 ordini dei dati) x 2 arm, lr fisso
  for SEED in 0 1 2; do
    for ARM in orig rot; do
      DIR=models/original; [ $ARM = rot ] && DIR=models/rotated
      python src/train.py --model-dir $DIR --arm $ARM \
          --optimizer adamw --lr 3e-4 --seed $SEED --steps $STEPS --batch-size $BS
    done
  done
  python src/analyze.py
  ;;
3)
  # sweep di learning rate, seed fisso
  for LR in 1e-4 2e-4 3e-4 6e-4 1e-3; do
    for ARM in orig rot; do
      DIR=models/original; [ $ARM = rot ] && DIR=models/rotated
      python src/train.py --model-dir $DIR --arm $ARM \
          --optimizer adamw --lr $LR --seed 0 --steps 300 --batch-size $BS
    done
  done
  python src/analyze.py
  ;;
plot)
  python src/analyze.py
  ;;
*)
  echo "uso: bash run_phases.sh {0|1|2|3|plot}"
  ;;
esac
