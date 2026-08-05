# kurtosis.py — misura quanto la base originale e' "privilegiata"
import torch, glob
from transformers import AutoModelForCausalLM

def kurt(x):                      # curtosi di Pearson: gaussiana = 3
    x = x.float().flatten()
    x = (x - x.mean()) / (x.std() + 1e-9)
    return (x**4).mean().item()

for tag, path in [("orig", "models/original"), ("rot", "models/rotated")]:
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    ks = []
    for layer in m.model.layers:
        for proj in (layer.self_attn.q_proj, layer.mlp.up_proj):
            W = proj.weight            # (out, d): colonne = dim del residual
            col_norms = W.norm(dim=0)  # energia per dimensione del residual
            ks.append(kurt(col_norms))
    t = torch.tensor(ks)
    print(f"{tag}: curtosi delle norme di colonna  media={t.mean():.2f}  max={t.max():.2f}")