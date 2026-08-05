"""
verify.py
=========
Il controllo di correttezza della Fase 0: i due gemelli DEVONO produrre
logit identici a meno dell'errore numerico fp32.

Se max_rel_err > 1e-3 c'e' un bug nella rotazione: NON proseguire con il training.
Atteso in fp32 con algebra fp64: ~1e-5 o meglio.

Uso:
  python src/verify.py --original models/original --rotated models/rotated \
      [--device cuda] [--n 64]
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "In the beginning was the",
    "def fibonacci(n):",
    "Il gatto sale sul",
    "E = mc^2 means that",
    "Once upon a time, a small",
    "1, 1, 2, 3, 5, 8,",
    "The gradient of a scalar function is a",
]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", default="models/original")
    p.add_argument("--rotated", default="models/rotated")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n", type=int, default=64, help="ripetizioni dei prompt (piu' token)")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.original)
    a = AutoModelForCausalLM.from_pretrained(args.original, dtype=torch.float32).to(args.device).eval()
    b = AutoModelForCausalLM.from_pretrained(args.rotated, dtype=torch.float32).to(args.device).eval()

    worst = 0.0
    for k in range(args.n):
        text = PROMPTS[k % len(PROMPTS)] + " " + "and so on," * (k // len(PROMPTS))
        ids = tok(text, return_tensors="pt").input_ids.to(args.device)
        la = a(ids).logits
        lb = b(ids).logits
        # errore relativo sui logit, normalizzato sulla scala dei logit
        rel = (la - lb).abs().max().item() / (la.abs().max().item() + 1e-9)
        worst = max(worst, rel)
        # controllo anche l'argmax (la predizione deve essere identica)
        same = (la.argmax(-1) == lb.argmax(-1)).float().mean().item()
        if k < 4:
            print(f"  prompt {k}: rel_err={rel:.3e}  argmax match={same:.4f}")

    print(f"\nmax errore relativo sui logit su {args.n} input: {worst:.3e}")
    if worst < 1e-3:
        print("PASS: i gemelli sono funzionalmente identici. Puoi passare alla Fase 1.")
    else:
        print("FAIL: errore troppo alto -> bug nella rotazione. NON accendere run di training.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
