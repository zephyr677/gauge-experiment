# Your LLM has a privileged coordinate system, and Adam is counting on it

**TL;DR.** I took a pretrained LLM (SmolLM2-135M), rewrote it in rotated
coordinates — *exactly* the same function, verified to numerical precision —
and fine-tuned both copies with identical data, identical batch order,
identical everything. Under SGD the two loss curves coincide to 1e-4, as
theory demands. Under AdamW they diverge hard: the rotated model's usable
learning-rate window shrinks by ~3×, with a gap of up to 1.0 nats at the
highest stable LR. The mechanism: the native basis of an Adam-pretrained model
approximately diagonalizes the Fisher information of the task, and Adam's
diagonal preconditioner is only a good natural-gradient approximation in a
basis where that holds. Rotating to the Fisher eigenbasis of a *broken*
(randomly rotated) model repairs ~75% of the damage — at zero runtime cost and
without knowing the original rotation. A continuous dose-response experiment
(rotation strength as a dial, function preserved at every point) then
head-to-heads three candidate predictors of the damage; the update-orthogonality
metric proposed by Maes, Zhang et al. wins, and the Fisher eigenbasis turns
out to be a *constructive procedure* for optimizing it.

All experiments run on a single L4 GPU for under $15 of compute. Code,
figures, and CSVs in this repo.

---

## 1. The symmetry

A transformer's residual stream has no privileged coordinate system *by
architecture*. For any orthogonal matrix Q you can rewrite every weight
matrix that reads from the residual stream (W_Q, W_K, W_V, MLP up/gate,
unembedding) as W → WQ, every matrix that writes to it (attention output,
MLP down) as W → QᵀW, and the embedding as W_E → W_E Q. After absorbing the
RMSNorm gains into adjacent weights, the resulting model computes the *same
function*, logit for logit. This is the same "computational invariance" that
QuaRot and SpinQuant exploit for 4-bit quantization.

So the coordinates are, in principle, a gauge freedom. The question: does the
*optimizer* care?

On paper the answer is known. SGD is equivariant: (W − η∇)Q = WQ − η∇Q, so
training the rotated twin is exactly the rotation of training the original.
Global-norm grad clipping and decoupled weight decay are equivariant too.
AdamW is not: its second-moment estimate is elementwise, and
(∇Q)⊙(∇Q) ≠ (∇⊙∇)Q. The magnitude of the effect was measured for
*pretraining from scratch* by Maes, Zhang et al. (2024) — see §5. What this
project measures is the complementary regime: a *pretrained* checkpoint,
whose basis was itself carved by Adam, rotated by an exactly
function-preserving gauge — which is what makes the SGD oracle and the causal
twin comparison below possible.

## 2. Setup

- **Model:** SmolLM2-135M (Llama-style: RMSNorm, RoPE, GQA, tied embeddings —
  untied before norm absorption). Full fine-tuning, fp32.
- **Twins:** `orig` = pretrained weights with RMSNorm gains absorbed;
  `rot` = the same, rotated by a random orthogonal Q (d = 576). Both
  constructed in float64 and verified: max relative logit deviation ~1e-5
  across 64 inputs.
- **Data:** wikitext-2, tokenized once and cached; the data-order seed is
  shared between twins, so seed variance measures noise while the twin gap
  measures the effect. Held-out eval set of 32 fixed blocks.
- **Determinism:** `torch.use_deterministic_algorithms`, fixed CUBLAS
  workspace, fp32 throughout.

## 3. Results

### 3.1 The oracle: SGD control

400 steps of SGD, same LR, both twins: |Δloss| averages **6e-5**, max
**1.7e-4** (fig1). This is the accumulated fp32 rounding error and nothing
else. It certifies the pipeline end-to-end — any AdamW gap above ~1e-3 is
signal, not bug. Most ML experiments have no correctness oracle; this design
gets one for free from the symmetry itself.

### 3.2 AdamW, cold start: a violent transient

Same protocol with AdamW (lr 3e-4, 3 data seeds per arm): the rotated twin's
loss explodes from 3.0 to ~7–8 within the first ~25 steps, then slowly
recovers to a persistent ~0.3 gap (fig2). The early blowup is Adam's cold
start: with empty second moments the first updates are approximately
±η·sign(g), and sign(gQ) ≠ sign(g)Q — the *sign pattern* of the gradient is
basis-dependent, so the initial perturbation differs violently between bases.

### 3.3 AdamW with warmup: the transient dies, the gap survives

Adding 50 warmup steps kills the explosion entirely but leaves a stable
~0.35 eval-loss gap from step ~75 onward (fig4, seed bands non-overlapping).
The effect is not just Adam-at-cold-start; it is steady-state.

### 3.4 The dose-response curve (main result)

LR sweep with warmup, tail-mean training loss over the last 50 steps:

| LR    | original | rotated | gap  |
|-------|----------|---------|------|
| 1e-5  | 2.76     | 2.76    | ~0   |
| 3e-5  | 2.73     | 2.77    | 0.04 |
| 1e-4  | 2.78     | 2.92    | 0.14 |
| 3e-4  | 3.11     | 4.12    | 1.01 |

The rotation does not move the optimum; it **shrinks the window**. The
maximum usable LR drops ~3×. This is the signature of a conditioning
problem: same function, different coordinates, and the adequacy of a
diagonal preconditioner degrades with the basis. (Replication: a second,
independent random rotation lands at 2.918 vs. the first rotation's 2.92 at
lr 1e-4 — the effect is a property of *changing basis*, not of one unlucky Q.)

### 3.5 Mechanism, part 1: outlier dimensions

Kurtosis of per-column weight norms (Gaussian = 3):

| model            | mean | max   |
|------------------|------|-------|
| original         | 17.7 | 120.2 |
| randomly rotated | 3.4  | 8.3   |

The native basis is violently non-Gaussian — a few residual dimensions carry
enormous energy (the well-known outlier dimensions, the same structure QuaRot
must destroy to quantize). A random rotation gaussianizes them.

### 3.6 Mechanism, part 2: the native basis diagonalizes the task Fisher

Attempted method: estimate the d×d residual-stream block of the task Fisher
(accumulate GᵀG over readers and GGᵀ over writers, 64 batches), rotate the
model into its eigenbasis Q*, and fine-tune. If Adam's diagonal
preconditioner is the bottleneck, the Fisher eigenbasis should be the best
possible coordinates.

Result: **Q\* exactly matches the native basis at every LR** — including
3e-4, where the random rotation collapses. And crucially, Q* is a *dense*
rotation (mean column peak 0.19, participation ratio 130/576), not a
permutation: it scrambles coordinates as thoroughly as the random Q does,
yet lands on native-level performance, and the resulting model is again
outlier-concentrated (kurtosis 15.4, max 177).

The conclusion writes itself: what Adam needs is not "the standard axes" —
it is **any basis that diagonalizes the task Fisher**. The native basis of an
Adam-pretrained checkpoint already is one, to excellent approximation. The
coordinate system of your model is not a convention; it is a fossil record of
the optimizer that created it. (This also gives a causal account of the
"privileged basis" phenomenon: architecture is rotation-invariant, but
elementwise Adam breaks the symmetry and carves axis-aligned structure that
it then depends on.)

As an accelerator, therefore, the method is useless on a healthy checkpoint:
there is nothing left to diagonalize. Honest negative.

### 3.7 Repair: recovering broken coordinates

But the method has a real use case: **repairing** a model whose coordinates
are bad. Take the randomly rotated twin (the 4.12 collapse), estimate *its*
Fisher, rotate into *its* eigenbasis, fine-tune at lr 3e-4:

| model                  | tail loss @ 3e-4 |
|------------------------|------------------|
| original               | 3.11             |
| randomly rotated       | 4.12             |
| rotated → Fisher-repaired | **3.37**      |

~75% of the damage recovered, at zero runtime cost, without any knowledge of
the original rotation — purely from the gradients of the broken model. The
residual 25% is consistent with eigenbasis estimation error (64 batches; the
bulk of the spectrum has near-degenerate eigenvalues whose eigenvectors are
noisy) and with the single-global-Q constraint, not with a limit of the
approach. Models that *live* in scrambled coordinates exist in the wild:
QuaRot/SpinQuant checkpoints fine-tuned for quantization are literally the
red curve.

### 3.8 Continuous dose-response, and a three-metric head-to-head

The experiments above compare discrete bases. To turn rotation strength into
a **continuous dial**, take a random antisymmetric A (rescaled to spectral
norm π) and set Q(t) = exp(tA): orthogonal — hence exactly
function-preserving, logit-checked — for *every* t ∈ [0, 1]. Training each
Q(t) twin (AdamW, lr 3e-4, warmup) gives a smooth dose-response with the
oracle intact at every point:

| t    | 0 | 0.25 | 0.5  | 0.75 | 1.0  | (Haar, ref.) |
|------|---|------|------|------|------|--------------|
| gap  | 0 | 0.09 | 0.19 | 0.25 | 0.24 | 1.01         |

Damage grows smoothly — no threshold, no cliff. Note that Q(1) is *not* as
strong as a Haar-random rotation (the exp family has few large angles), and
correspondingly does far less damage: the dose-response continues well beyond
t = 1 toward the Haar point.

These 7 bases (native, four Q(t), Haar, Q*) then serve as a test bed for
**three candidate predictors of the damage**, each computable without
training:

| predictor | source | Spearman ρ with gap | verdict |
|---|---|---|---|
| L1 off-diagonal mass of the rotated Fisher block | this project, first attempt | +0.96 (inflated) | **fails where it matters**: assigns Haar and Q(1) values identical to the 4th decimal (0.99458 vs 0.99457) while their damages differ 4× (1.01 vs 0.24). The high ρ is carried almost entirely by the Q* point at zero. Descriptive, not predictive. |
| log-condition-number of the diagonally-preconditioned Fisher | this project, second attempt | +0.46 | **fails outright**: non-monotone even inside the Q(t) family. λ_min of a 64-batch Fisher estimate is sampling noise (including the near-null gauge directions), and κ is dominated by it. A theoretically motivated but statistically fragile proposal — disclosed as such. |
| coefficient of variation of the singular values of Adam's effective update m̂/(√v̂+ε), per layer | Maes, Zhang et al. (2024), their §4.2 | +0.96 | **wins**: orders all 7 points correctly (the single t=0.75/t=1 inversion is within single-seed noise), *separates* Haar (CV 1.399) from Q(1) (1.313) where the L1 metric saturates, and places Q* (0.948) below even the native basis (1.099). |

The last number is the punchline of the comparison:

> **CV(Q\*) = 0.948 < CV(native) = 1.099.** The Fisher eigenbasis produces
> Adam updates *more orthogonal than the native basis itself*. This welds the
> two candidate quantities into one story: diagonalizing the residual Fisher
> block ⟹ decorrelated gradients along the residual index ⟹ sane
> per-coordinate scale estimates in v̂ ⟹ more uniform update singular values.
> Their metric measures how good a basis is; the Fisher eigenbasis of §3.6 is
> a **constructive procedure** for manufacturing a good one from gradients
> alone. Their metric, our knob.

**Falsification test.** If CV is the causal quantity, Q* — with a *better* CV
than native — should tolerate a slightly higher LR. Tested at lr 6e-4
(single seed): native tail 3.56, Q* tail 3.61; both degrade (eval above the
starting point — 6e-4 is outside the window for both); Q* is somewhat more
stable early (max training-loss spike 4.09 vs 4.48) but ends nominally worse.
**Verdict: parity within single-seed noise.** The CV advantage does not
translate into a measurably wider window at this resolution — the metric
ranks bad bases well, but does not discriminate between two good ones. Both
outcomes were pre-registered; this is the honest one.

## 4. What this does NOT show

- **No speedup for normal fine-tuning.** The native basis is already
  Fisher-diagonalizing for this task; rotating a healthy checkpoint buys
  nothing. If your coordinates are fine, leave them alone. The 6e-4 test in
  §3.8 confirms this from the other side: even a basis with a *better*
  update-orthogonality score than native does not beat native.
- **One model (135M), one task (wikitext-2), short runs (400 steps).** The
  qualitative mechanism should transfer (the symmetry and Adam's elementwise
  structure are universal), but magnitudes may not.
- **Single-seed points.** The Q(t) sweep and the 6e-4 test are one seed each;
  the seed band at 3e-4 is roughly ±0.1, so differences below that are not
  interpreted anywhere in this document.
- **Fine-tuning only.** Pretraining-from-scratch dynamics, where the basis is
  *being formed* rather than inherited, are untouched here — and are arguably
  the more interesting regime.
- **The Fisher block is layer-averaged.** The gauge constraint forces a
  single global Q; per-layer optimal bases exist but would break functional
  invariance. This is a structural limit of any gauge-respecting method.

## 5. Relation to prior work

This is the section that changed most since the first draft of this README —
a literature pass turned up work that is much closer to this result than I
initially realized, and it changes how the project should be framed. Below,
roughly in order of how directly they overlap with the core claim.

- **Maes, Zhang et al., "Understanding Adam Requires Better Rotation
  Dependent Assumptions" (2024, arXiv:2410.19964).** This is the closest
  prior work, not an adjacent one. They formally prove SGD's rotation
  equivariance and Adam's lack of it, and empirically show Adam's training
  loss on GPT-2 and ViT degrades under random parameter-space rotations, with
  degradation scaling with rotation scope (output-wise < input-wise <
  layer-wise < global) — the same dose-response shape as our §3.4 sweep, on
  different architectures and a different measurement (training-loss curves
  under fixed LR, vs. our LR-window sweep). Crucially, they also test
  ResNet-50 and find *no* meaningful degradation under rotation — direct
  outside confirmation that the effect is Transformer/Adam specific, not a
  generic optimizer artifact. Their rotations act on the *gradients inside
  the optimizer* (rotated-Adam on a fixed network); ours act on the *model*
  (plain AdamW on function-identical twins) — two complementary views of the
  same symmetry breaking, and only the latter admits the SGD oracle.
  Their SVD-rotation result (their Fig. 5) also sharpens our §4 limitation
  from the other side: a per-matrix, periodically recomputed eigenbasis
  rotation *improves over* the native basis, while our gauge-constrained
  single global Q* only *matches* it — evidence that the layer-averaging
  forced by functional invariance is the binding constraint, not the
  Fisher-eigenbasis idea itself.

  Where this project adds something concrete: their §4.1 tests existing
  rotation-dependent assumptions (L∞-gradient bounds; Hessian
  block-diagonality; the (1,1)-Hessian-norm probe of Xie et al., 2025) and
  finds each falls short — the (1,1)-norm correlates under global and SVD
  rotations but breaks under output-wise ones. Their §4.2 then proposes
  **update orthogonality** as a promising indicator, and their conclusion
  calls for formalizing it. **§3.8 of this project is a direct test of that
  proposal on an independent test bed** (7 function-identical bases of a
  pretrained checkpoint, including two the one-parameter family cannot
  produce): their metric wins against two Fisher-based alternatives,
  including this project's own first two attempts. And §3.6–3.7 supply what a
  metric alone cannot: a *constructive* procedure (the Fisher eigenbasis)
  that manufactures high-orthogonality bases from gradients alone, repairing
  75% of the damage of an unknown rotation. A response/extension to their
  explicitly-stated open problem, not a parallel finding.

- **Sareen, Pedramfar, Kaba, Shakerinava, Ravanbakhsh, "The Role of Symmetry
  in Optimizing Overparameterized Networks" (Mila/McGill, April 2026,
  arXiv:2604.25150).** Independent line of work on a *different* symmetry
  (neuron-splitting/duplication under overparameterization, not O(d)
  residual-stream rotation), but converges on the same underlying picture:
  symmetry transformations act as *diagonal preconditioning* on the Hessian
  (their central preconditioning theorem), and the benefit is contingent on
  how axis-aligned the true curvature already is. Their Appendix H is the
  most directly relevant part: they measure Hessian eigenvector participation
  ratios in pretrained Transformers and CNNs and find **empirical
  axis-alignment specifically in Transformers**, then cite Maes/Zhang as the
  explanation for why Adam benefits from it and underperforms under rotation
  in Transformers but not ResNets — i.e. they treat our §3.4–3.5 phenomenon
  as an established fact and build on top of it rather than re-deriving it.
  Useful as a second, independent confirmation that "axis-alignment +
  diagonal preconditioner" is a live research thread in 2026, and as a source
  of the diagonal-preconditioning formalism that could sharpen the informal
  argument here.

- **Dong & Cheng, "Quotient Geometry, Effective Curvature, and Implicit Bias
  in Simple Shallow Neural Networks" (2026, arXiv:2603.21502).** Purely
  theoretical, shallow networks only (quadratic/ReLU activation, no
  Transformers, no Adam) — but supplies rigorous vocabulary for something
  this project currently argues informally. They formally separate "false
  flatness" (Hessian zero-eigenvalues from symmetry orbits — pure
  reparametrization artifact) from "intrinsic flatness" (genuine curvature
  of the predictor after quotienting out the symmetry group), and build a
  metric on the quotient manifold via pullback of the realization map's
  Jacobian — essentially a Gauss-Newton/empirical-Fisher construction. Their
  vertical/horizontal decomposition of gradient flow (vertical = pure gauge
  motion, invisible to the loss; horizontal = the only component that moves
  the predictor) is a clean formal frame for the O(d) residual-stream gauge
  used here: our rotations are exactly vertical directions of an analogous
  symmetry group acting on a much bigger, non-shallow model.

- **Concurrent:** "The Loss Does Not See the Basis, but Adam Does"
  (arXiv:2608.05136, Aug 2026) studies the *implicit-bias* face of the same
  coin — which interpolant Adam selects in factored models at matched
  training loss, as a function of basis — versus the *training-dynamics*
  face measured here (LR window, damage, repair). Same title thesis,
  different question; no overlap in results, and further evidence this
  thread is active right now.

- **QuaRot / SpinQuant** exploit the same computational invariance for
  inference (outlier smoothing for 4-bit quantization). This project uses it
  as an experimental scalpel for *training*.
- **SOAP** (Vyas et al., ICLR 2025) runs Adam in the Shampoo preconditioner's
  eigenbasis — the dynamic, per-matrix, runtime-cost version of the same
  geometric idea. Its 40% iteration savings in pretraining is indirect
  evidence for the phenomenon measured here; the present result is the
  static, global, zero-runtime-cost, function-preserving slice of it, plus
  the causal measurement SOAP doesn't do (SOAP never compares
  function-identical twins). The optimizer thread has since moved further:
  ARO (Gong et al., 2026) unifies eigen-rotation/SOAP/Muon schemes with
  1.3×+ step-speedups at LLM scale, and PolarAdamW (arXiv:2605.07067)
  studies gauge-equivariance in matrix optimizers — both consistent with the
  picture here that the basis, not the architecture, is where Adam's
  advantage lives.
- **Riemannian-preconditioned LoRA** (Zhang & Pilanci, ICML 2024) and
  **conservation-law analyses** (Kunin et al., "Neural Mechanics") cover the
  GL(r) symmetry of low-rank factorizations; the O(d) residual-stream gauge
  studied here is the full-model analogue.
- **Privileged basis** discussions in interpretability (Elhage et al.) ask
  why residual streams have axis-aligned features despite rotation-invariant
  architecture. The "Adam *creates* it" direction has been shown directly:
  Caples & Neuhaus (LessWrong, "Adam Optimizer Causes Privileged Basis in
  Transformer Language Models") train from scratch with Adam vs. SGD and
  measure excess kurtosis — the same metric as §3.5, from the opposite side.
  What the experiments here add is the *converse* causal direction: Adam also
  **depends on** the basis it carved — removing it (while preserving the
  function exactly) shrinks the usable LR window 3×. Creation and dependence
  together close the loop.

## 6. Reproduce

```
pip install -r requirements.txt
bash run_phases.sh 0             # build twins + verify logits (CPU, ~10 min)
bash run_phases.sh 1             # SGD oracle (~30 min L4)
bash run_phases.sh 2             # AdamW, 3 seeds (~2.5 h L4)
bash run_phases.sh 3             # LR sweep (~3 h L4)
python src/fisher_basis.py       # + rotate_model.py --q-file for §3.6–3.7
python src/qt_sweep.py --stage build|fisher|train|orth|analyze   # §3.8
python src/qt_extra.py --stage metrics|cv|scatter                # §3.8 head-to-head
```

Total compute: ~15 credits (~$15) on a Lightning AI L4. Every number in this
README comes out of a CSV in `results*/`.

---

*Questions, replications, and "you missed this paper" comments welcome —
especially the last kind.*
