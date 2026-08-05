"""
rotate_model.py
===============
Costruisce il "gemello ruotato" di un modello Llama-style (SmolLM2, Llama 3.2, Qwen2...):

  1. slega lm_head da embed_tokens (se tied)
  2. assorbe i gain di tutte le RMSNorm nelle matrici adiacenti
  3. applica una rotazione ortogonale globale Q al residual stream:
        - chi LEGGE dal residual  (q,k,v, gate, up, lm_head):  W -> W @ Q
        - chi SCRIVE nel residual (o_proj, down_proj):         W -> Q^T @ W
        - embedding:                                           W_E -> W_E @ Q
  4. salva il modello ruotato (stessa architettura, stessa funzione, coordinate diverse)

Tutta l'algebra e' fatta in float64 e ricastata al dtype originale.

Uso:
  python src/rotate_model.py --model HuggingFaceTB/SmolLM2-135M \
      --out-original models/original --out-rotated models/rotated --seed 1234
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def untie_lm_head(model):
    """Slega lm_head dall'embedding (necessario prima di assorbire la norm finale)."""
    tied = getattr(model.config, "tie_word_embeddings", False)
    if tied or model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(
            model.lm_head.weight.detach().clone()
        )
        model.config.tie_word_embeddings = False
        print("  [untie] lm_head scollegato da embed_tokens")
    return model


@torch.no_grad()
def absorb_rmsnorm(model):
    """
    Assorbe i gain gamma delle RMSNorm nelle matrici a valle e li mette a 1.
    Dopo questo passo la RMSNorm e' pura normalizzazione -> equivariante per rotazioni.

    y = W @ (g * x_norm)  ==>  y = (W * g) @ x_norm      (g scala le COLONNE di W)
    """
    for i, layer in enumerate(model.model.layers):
        # input_layernorm -> q, k, v
        g = layer.input_layernorm.weight.double()
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            proj.weight.copy_((proj.weight.double() * g).to(proj.weight.dtype))
        layer.input_layernorm.weight.fill_(1.0)

        # post_attention_layernorm -> gate, up
        g = layer.post_attention_layernorm.weight.double()
        for proj in (layer.mlp.gate_proj, layer.mlp.up_proj):
            proj.weight.copy_((proj.weight.double() * g).to(proj.weight.dtype))
        layer.post_attention_layernorm.weight.fill_(1.0)

    # norm finale -> lm_head
    g = model.model.norm.weight.double()
    model.lm_head.weight.copy_(
        (model.lm_head.weight.double() * g).to(model.lm_head.weight.dtype)
    )
    model.model.norm.weight.fill_(1.0)
    print("  [absorb] gain RMSNorm assorbiti (tutti i gamma ora = 1)")
    return model


def random_orthogonal(d, seed):
    """Q ortogonale d x d in float64 via QR di una gaussiana. cond(Q) = 1."""
    gen = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=gen, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    # fissa i segni per unicita' (diagonale di R positiva)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    err = (Q.T @ Q - torch.eye(d, dtype=torch.float64)).abs().max().item()
    print(f"  [Q] ortogonalita': max|Q^T Q - I| = {err:.2e}")
    return Q


@torch.no_grad()
def rotate(model, Q):
    """Applica il cambio di gauge x -> Q^T x al residual stream."""
    Qd = Q.double()

    def read(proj):   # W (out, d) legge dal residual: W -> W Q
        proj.weight.copy_((proj.weight.double() @ Qd).to(proj.weight.dtype))

    def write(proj):  # W (d, out) scrive nel residual: W -> Q^T W
        proj.weight.copy_((Qd.T @ proj.weight.double()).to(proj.weight.dtype))

    for layer in model.model.layers:
        read(layer.self_attn.q_proj)
        read(layer.self_attn.k_proj)
        read(layer.self_attn.v_proj)
        write(layer.self_attn.o_proj)
        read(layer.mlp.gate_proj)
        read(layer.mlp.up_proj)
        write(layer.mlp.down_proj)

    # embedding: le righe sono i vettori residui -> weight @ Q
    emb = model.model.embed_tokens
    emb.weight.copy_((emb.weight.double() @ Qd).to(emb.weight.dtype))
    # lm_head legge dal residual -> weight @ Q
    model.lm_head.weight.copy_(
        (model.lm_head.weight.double() @ Qd).to(model.lm_head.weight.dtype)
    )
    print("  [rotate] rotazione applicata a tutto lo stack")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--out-original", default="models/original")
    p.add_argument("--out-rotated", default="models/rotated")
    p.add_argument("--seed", type=int, default=1234, help="seed della rotazione Q")
    p.add_argument("--q-file", default=None, help="carica Q da file (es. Qstar.pt) invece di generarla casuale")
    p.add_argument("--skip-original", action="store_true", help="non risalvare l'arm originale")
    args = p.parse_args()

    print(f"Carico {args.model} in fp32...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    d = model.config.hidden_size
    print(f"  hidden_size = {d}, layers = {model.config.num_hidden_layers}")

    # --- Arm A: originale con norm assorbite (baseline equa!) ---
    # NB: assorbiamo le norm ANCHE nell'originale, cosi' i due arm differiscono
    # SOLO per la rotazione e per nient'altro.
    model = untie_lm_head(model)
    model = absorb_rmsnorm(model)
    if not args.skip_original:
        model.save_pretrained(args.out_original)
        tok.save_pretrained(args.out_original)
        print(f"Arm A (originale, norm assorbite) salvato in {args.out_original}\n")

    # --- Arm B: gemello ruotato ---
    if args.q_file:
        Q = torch.load(args.q_file).double()
        err = (Q.T @ Q - torch.eye(d, dtype=torch.float64)).abs().max().item()
        print(f"  [Q] caricata da {args.q_file}, ortogonalita': {err:.2e}")
        assert err < 1e-8, "Q non ortogonale!"
    else:
        Q = random_orthogonal(d, args.seed)
    model = rotate(model, Q)
    model.save_pretrained(args.out_rotated)
    tok.save_pretrained(args.out_rotated)
    torch.save(Q, f"{args.out_rotated}/Q.pt")
    print(f"Arm B (ruotato) salvato in {args.out_rotated}")
    print("\nOra esegui: python src/verify.py")


if __name__ == "__main__":
    main()
