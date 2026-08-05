"""
train.py
========
Un singolo run di fine-tuning deterministico. La stessa identica sequenza di
batch viene mostrata a entrambi gli arm (il seed controlla SOLO l'ordine dei dati).

Uso:
  python src/train.py --model-dir models/original --arm orig \
      --optimizer adamw --lr 3e-4 --seed 0 --steps 400 --outdir results

Output: results/{arm}_{optimizer}_lr{lr}_seed{seed}.csv  (colonne: step, loss)
"""

import os
# determinismo CUDA: DEVE stare prima dell'import di torch
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_dataset(tokenizer, block_size=512, cache="data_cache.pt"):
    """Tokenizza wikitext-2 una volta sola e la impacchetta in blocchi fissi.
    La cache garantisce che TUTTI i run vedano gli stessi identici tensori."""
    if os.path.exists(cache):
        return torch.load(cache)
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = []
    for ex in ds:
        t = ex["text"]
        if t.strip():
            ids.extend(tokenizer(t).input_ids)
    ids = torch.tensor(ids, dtype=torch.long)
    n_blocks = len(ids) // block_size
    blocks = ids[: n_blocks * block_size].view(n_blocks, block_size)
    torch.save(blocks, cache)
    print(f"[data] {n_blocks} blocchi da {block_size} token -> cache {cache}")
    return blocks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--arm", required=True, choices=["orig", "rot", "fisher"])
    p.add_argument("--optimizer", required=True, choices=["sgd", "adamw"])
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, default=0, help="controlla SOLO l'ordine dei dati")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0, help="grad clip globale (gauge-invariante); 0 = off")
    p.add_argument("--warmup", type=int, default=0, help="step di warmup lineare del LR")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    # determinismo totale
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32
    ).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    all_blocks = build_dataset(tok, args.block_size)
    eval_blocks = all_blocks[-32:]          # held-out FISSO, uguale per tutti i run
    blocks = all_blocks[:-32]

    # permutazione dei dati: dipende SOLO da --seed, identica per i due arm
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(blocks), generator=g)

    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(0.9, 0.95), eps=1e-8,
                                weight_decay=args.weight_decay)

    os.makedirs(args.outdir, exist_ok=True)
    name = f"{args.arm}_{args.optimizer}_lr{args.lr:g}_seed{args.seed}"
    path = os.path.join(args.outdir, name + ".csv")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["step", "loss", "eval_loss"])

    bs = args.batch_size
    for step in range(args.steps):
        idx = perm[(step * bs) % len(blocks): (step * bs) % len(blocks) + bs]
        if len(idx) < bs:  # wrap-around a fine epoca
            idx = torch.cat([idx, perm[: bs - len(idx)]])
        batch = blocks[idx].to(device)

        if args.warmup > 0 and step < args.warmup:
            for grp in opt.param_groups:
                grp["lr"] = args.lr * (step + 1) / args.warmup

        out = model(input_ids=batch, labels=batch)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        ev = ""
        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                tot = 0.0
                for j in range(0, len(eval_blocks), bs):
                    eb = eval_blocks[j:j+bs].to(device)
                    tot += model(input_ids=eb, labels=eb).loss.item() * len(eb)
                ev = f"{tot/len(eval_blocks):.6f}"
            model.train()
            print(f"[{name}] step {step:4d}  loss {loss.item():.4f}  eval {ev}")
            f.flush()
        w.writerow([step, f"{loss.item():.6f}", ev])

    f.close()
    print(f"[done] {path}")


if __name__ == "__main__":
    main()
