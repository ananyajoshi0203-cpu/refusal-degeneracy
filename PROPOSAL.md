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
4. **A rank bound on ablation robustness (novel).** Refusal retained as a function of the number
   of ablated input-conditional directions. The claim: a defense's robustness is bounded by its
   number of distinct refusal modes, and an adaptive attacker who re-estimates the directions on
   the defended model wins once the cluster count reaches it. Degeneracy raises the rank an
   attacker needs; it cannot remove the bound. The quantity reported per defended model is the
   rank needed to remove refusal under the KL budget.

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

Training against the attacker does not escape the bound (toy models, 3 seeds, 3 harm categories
each refusing with its own token). Training the FFN to keep refusing while its own current global
and cluster directions are ablated, re-estimated every 12 steps with the cluster count growing to
6, retains this much of the refusal gap against a fresh cluster attack with k directions:

| Seed | Training | k = 1 | k = 2 | k = 4 | k = 8 |
|---|---|---|---|---|---|
| 0 | plain refusal | 0.28 | 0.14 | -0.02 | -0.01 |
| 0 | against own subspace | 0.83 | 0.58 | 0.03 | 0.01 |
| 1 | plain refusal | 0.23 | 0.11 | 0.01 | 0.00 |
| 1 | against own subspace | 0.11 | -0.04 | -0.02 | 0.02 |
| 2 | plain refusal | 0.17 | 0.17 | -0.17 | -0.10 |
| 2 | against own subspace | 0.41 | 0.14 | -0.04 | -0.02 |

It can raise robustness at k below the number of refusal modes (two seeds of three), and
nothing survives once k reaches it. With a single refusal token for every harmful prompt,
refusal is removed at k = 1 whatever the training, since the final-layer class-mean difference
then always contains that token's unembedding.

Real model, undefended Qwen2.5-7B-Instruct (A100; refusal judged by string match until
Llama-Guard-3-8B access is granted). The refusal direction sits at layer 18, and a single MLP
neuron (layer 25, index 17058) pinned at 100 is the strongest neuron attack.

| Test set | Baseline refusal | Direction ablation | Neuron pin |
|---|---|---|---|
| StrongREJECT-313 | 0.82 | 0.03 | 0.14 |
| JailbreakBench-100 | 0.91 | 0.02 | 0.17 |

Direction ablation leaves 304/313 StrongREJECT prompts answered, against DSA Table V's 299/313
unsafe for the same model, so the undefended row reproduces. Two findings from the same run
change the plan:

- **The first-token refusal score is not a faithful proxy for generated refusal.** Under direction
  ablation it keeps 0.50 of its harmful-vs-harmless gap while generated refusal falls from 0.82 to
  0.03. Survival, the counterfactual share and the cluster sweep are all scored on that gap, so
  on real models they must be scored on generated refusal over validation prompts instead.
- **Cluster directions cost far more harmless KL than random ones** (0.22 at k = 2 and 3.4 at
  k = 8, against 0.06-0.19 for the random control), so no cluster count fits the 0.1 KL budget on
  the undefended model. The single direction already removes refusal there, so this matters only
  for defended models; the cluster directions should be orthogonalized against the harmless
  activation mean before ablation.
- The undefended model's most attributed safety neurons barely write along the direction
  (counterfactual share 0.001-0.02 for the top 1-100), as expected with no defense installed.

Reimplemented defenses (no code was released for either): DSA (`refusal_degeneracy/dsa.py`;
training only the FFN reproduces the paper's stated 74.88% of trained parameters as 75%) and
NeuronGuard (`refusal_degeneracy/neuronguard.py`).

## Experiments

| Model | Defense | Attacks | Measures |
|---|---|---|---|
| Qwen2.5-7B-Instruct | none, DSA, NeuronGuard | single neuron, single direction, cluster (k = 2-16), random control | ASR (Llama-Guard-3-8B, StrongREJECT-313, JBB-100), counterfactual share, rank needed under the KL budget, KL |
| LLaMA-3.2-3B-Instruct | none, DSA | same | same |
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
- **Proxy metric**: see above; any survival number scored on the first-token gap is suspect
  until it is rescored on generations.
- **DSA reimplementation fidelity**: the paper leaves the refusal targets, data pairing and
  precision unstated. The undefended and DSA rows of Table V are the check.
- **Gated models**: Llama-Guard-3-8B and Llama-3.2 need the Meta license accepted on the
  Hugging Face account.
- **Compute**: an A100 costs 8.88 Colab units per hour, and about 84 units are left. Per-run
  times are being measured on the first Qwen2.5-7B chain; NeuronGuard at batch size 1 is about
  20k steps and should be batched before it is run.
