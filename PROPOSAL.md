# Redundant or Degenerate? Predicting Which Neuron-Level Safety Defenses Survive Refusal-Direction Ablation

Target venue: NeurIPS 2027 (main track).

## The gap

Neuron-level safety defenses spread refusal across many neurons so that no single neuron is a
point of failure: DSA (Xie et al. 2026), NeuronGuard (Gao et al. 2026), SafeNeuron, SN-Tune.
They are evaluated against neuron attacks (pruning, pinning, Kazemi et al. 2026). The classic
representation attack, ablating the difference-in-means refusal direction (Arditi et al. 2024),
gives contradictory results on them, and nobody can say in advance which way it will go:

- DSA Table V: ablation leaves DSA-Qwen2.5-7B almost fully jailbroken (289/313 unsafe, against
  299/313 undefended) while DSA-LLaMA-3.2-3B holds (17/313, against 133/313 undefended). The
  paper reports the split and does not explain it.
- NeuronGuard is never tested against direction ablation at all.
- No neuron-level defense has been tested against multi-direction ablation (concept cones,
  Wollschläger et al. 2025; SOM directions, Piras et al. 2025), which breaks circuit breakers.

## The idea

Spreading refusal over many neurons can mean two different things, and only one of them helps.

- **Redundancy**: many neurons that write (nearly) the same direction. Removing a neuron leaves
  its copies, so neuron attacks fail. But the copies share one direction, and ablating that
  direction removes all of them at once.
- **Degeneracy**: structurally different neurons that produce refusal through different
  directions. Ablating one direction leaves the others.

Two refinements make this precise.

1. **Degeneracy must be input-conditional.** If diverse neurons all fire on every harmful prompt,
   their writes sum to one mean write, and the difference-in-means direction is exactly that
   sum. Ablating it removes them all, however different their individual writes are (a short
   lemma from linearity of the mean). A defense survives single-direction ablation only when
   different harmful inputs recruit different neurons.
2. **Input-conditional degeneracy is still low-rank.** If refusal runs through a few
   input-specific directions, an attacker who ablates one direction per cluster of harmful
   activations removes them all. So the real robustness quantity is how many directions an
   attacker must remove, not whether one suffices.

## Contributions

1. **A predictor (novel).** The counterfactual direction share of a defense's safety neurons:
   the drop in the refusal gap when only the component of their writes along the refusal
   direction is removed, over the drop when their writes are removed entirely. It is computed
   on validation prompts before any attack is run, and predicts whether the defense survives.
2. **An explanation of DSA's Table V split (novel).** Measure the share on DSA-Qwen2.5-7B and
   DSA-LLaMA-3.2-3B; the prediction is a high share (redundant) for Qwen and a low share
   (degenerate) for LLaMA.
3. **The first multi-direction evaluation of neuron-level defenses (partially novel; the attack
   is SOM's, with k-means).** Cluster-conditional ablation against DSA and NeuronGuard, under
   the same KL budget as the single-direction attack and with a random-direction control at
   every cluster count.
4. **A rank-robustness curve and a defense that moves it (novel).** Refusal retained as a
   function of the number of ablated input-conditional directions. Degenerate Safety Training
   trains refusal to survive ablation of a growing set of the model's own cluster-conditional
   directions, and is evaluated against single-direction, cluster, SOM and cone attacks, and an
   adaptive attacker who knows the training clusters.

Closest prior work, to be addressed directly: Piras et al. 2025 (SOM directions; the attack in
3 is theirs), Wollschläger et al. 2025 (concept cones), ReFAT (Yu et al. 2025; trains against a
single direction), refusal aliases (Truong 2026; fixed, input-independent aliases), and two
results that refusal is shared downstream even when harm is category-specific (2507.11878,
2602.02132). That last point is the main risk to contribution 2; see Risks.

## What is already established

Synthetic ground truth (CPU, `scripts/run_synthetic.py`): 96 defenses implanted into small
random Llamas, sweeping how degenerate the per-category safety neurons' writes are.

- The counterfactual share predicts single-direction survival at Spearman -0.92. The
  gradient-based direction share manages -0.30 and plain weight geometry (weighted cos^2)
  -0.42: gradient x activation even gives neurons writing exactly along the direction a
  negative attribution, because the final RMSNorm saturates.
- Mean share of the refusal gap retained, per degeneracy level:

  | Degeneracy | Single direction | Global + cluster directions | Global + random (control) |
  |---|---|---|---|
  | 0.00 (redundant) | -0.00 | -0.02 | 0.01 |
  | 0.25 | 1.04 | 0.11 | 0.98 |
  | 0.50 | 0.87 | 0.00 | 0.83 |
  | 1.00 (degenerate) | 0.71 | -0.01 | 0.68 |

  Degenerate defenses survive the single direction and fall to the cluster attack; random
  directions of the same rank leave refusal intact.

Real model, undefended Qwen2.5-7B-Instruct (A100, in progress): the refusal direction sits at
layer 18; a single MLP neuron (layer 25, index 17058) pinned at 100 drops the validation
refusal score from 4.2 to -3.0 at KL 0.03, reproducing Kazemi et al.'s single-neuron attack on
this model. Baseline JailbreakBench refusal rate is 0.91.

Reimplemented defenses (no code was released for either): DSA (`refusal_degeneracy/dsa.py`;
training only the FFN reproduces the paper's stated 74.88% of trained parameters as 75%) and
NeuronGuard (`refusal_degeneracy/neuronguard.py`).

## Experiments

| Model | Defense | Attacks | Measures |
|---|---|---|---|
| Qwen2.5-7B-Instruct | none, DSA, NeuronGuard, DST | single neuron, single direction, cluster (k = 2-16), random control | ASR (Llama-Guard-3-8B, StrongREJECT-313, JBB-100), counterfactual share, rank-robustness curve, KL |
| LLaMA-3.2-3B-Instruct | none, DSA, DST | same | same |
| LLaMA-3.1-8B-Instruct | none, NeuronGuard | same | same, for a third point on the predictor |

The predictor claim needs more than two defended models to be credible: score every
(model, defense) pair and report the rank correlation of the share with measured survival.

Matching DSA's setup (StrongREJECT-313, Llama-Guard-3-8B) lets the undefended and DSA rows be
checked against their Table V before any new number is trusted.

## Risks

- **Refusal may merge downstream** (2507.11878, 2602.02132): if input-specific harm features
  converge on one refusal direction by the attacked layer, degeneracy upstream does not protect.
  The share is measured at the attacked layer, so this shows up as a high share rather than a
  wrong prediction; the paper should report the share across layers.
- **DSA reimplementation fidelity**: the paper leaves the refusal targets, data pairing and
  precision unstated. The undefended and DSA rows of Table V are the check.
- **Gated models**: Llama-Guard-3-8B and Llama-3.2 need the Meta license accepted on the
  Hugging Face account.
- **Compute**: an A100 costs 8.88 Colab units per hour, and about 84 units are left. Per-run
  times are being measured on the first Qwen2.5-7B chain; NeuronGuard at batch size 1 is about
  20k steps and should be batched before it is run.
