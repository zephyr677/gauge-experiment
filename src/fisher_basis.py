"""
fisher_basis.py
===============
Stima il blocco d x d della Fisher relativo alle dimensioni del RESIDUAL STREAM
sul task di fine-tuning, e ne calcola l'autobase Q*.

Idea: per ogni matrice W che LEGGE dal residual (grad G di shape (out, d)),
il momento secondo del gradiente lungo l'indice residuale e' G^T G (d x d).
Per chi SCRIVE (grad G di shape (d, out)) e' G G^T. Sommiamo tutto su layer
e batch -> F. Se ruotiamo il modello con Q* = autobase di F, nel nuovo
sistema di coordinate F diventa diagonale: il precondizionatore DIAGONALE
di Adam vede dimensioni residuali decorrelate.

Uso:
  python src/fisher_basis.py --model-dir models/original --batches 64 --out Qstar.pt
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from train import build_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/original")
    p.add_argument("--batches", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0, help="stesso seed dati del training")
    p.add_argument("--out", default="Qstar.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32
    ).to(device)
    model.gradient_checkpointing_enable()
    model.train()
    d = model.config.hidden_size

    all_blocks = build_dataset(tok, args.block_size)
    blocks = all_blocks[:-32]                      # stesso split di train.py
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(blocks), generator=g)

    F = torch.zeros(d, d, dtype=torch.float64, device=device)
    bs = args.batch_size

    def readers_writers():
        for layer in model.model.layers:
            yield "r", layer.self_attn.q_proj
            yield "r", layer.self_attn.k_proj
            yield "r", layer.self_attn.v_proj
            yield "w", layer.self_attn.o_proj
            yield "r", layer.mlp.gate_proj
            yield "r", layer.mlp.up_proj
            yield "w", layer.mlp.down_proj
        yield "r", model.lm_head
        # embed_tokens: grad (vocab, d), righe = residui -> come reader
        yield "r", model.model.embed_tokens

    for b in range(args.batches):
        idx = perm[(b * bs) % len(blocks): (b * bs) % len(blocks) + bs]
        if len(idx) < bs:
            idx = torch.cat([idx, perm[: bs - len(idx)]])
        batch = blocks[idx].to(device)
        loss = model(input_ids=batch, labels=batch).loss
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for kind, mod in readers_writers():
                G = mod.weight.grad
                if G is None:
                    continue
                G = G.double()
                F += G.T @ G if kind == "r" else G @ G.T
        if b % 8 == 0:
            print(f"[fisher] batch {b}/{args.batches}  loss {loss.item():.4f}")

    model.zero_grad(set_to_none=True)
    F = F.cpu()
    F = 0.5 * (F + F.T)                            # simmetrizza (paranoia numerica)
    evals, evecs = torch.linalg.eigh(F)            # autobase, gia' ortogonale
    Q = evecs.flip(1)                              # autovalori decrescenti
    err = (Q.T @ Q - torch.eye(d, dtype=torch.float64)).abs().max().item()
    top = evals.flip(0)[:8] / evals.sum()
    print(f"\n[Q*] ortogonalita': {err:.2e}")
    print(f"[Q*] frazione di energia dei primi 8 autovalori: "
          + ", ".join(f"{v:.3f}" for v in top))
    print(f"[Q*] rapporto lambda_max/lambda_min: {evals.max()/evals.clamp_min(1e-20).min():.2e}")
    torch.save(Q, args.out)
    print(f"[Q*] salvata in {args.out}")
    print("Ora: python src/rotate_model.py --q-file Qstar.pt --out-rotated models/fisher ...")


if __name__ == "__main__":
    main()
