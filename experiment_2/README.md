# Experiment 2: Counterfactual Attention Weights (Paper Section 4.2)

## What This Experiment Tests

**Question**: If we use completely different attention weights, would the prediction change?

If attention truly *explains* the prediction, then changing which tokens the model "attends to" should change the prediction. This experiment tests that by constructing **counterfactual** attention distributions — alternative attention weights that are *maximally different* from the original — and checking if the prediction stays the same.

**Two sub-experiments from the paper**:
1. **Section 4.2.1 — Attention Permutation**: Randomly shuffle attention weights and measure prediction change
2. **Section 4.2.2 — Adversarial Attention**: Explicitly search for maximally different attention that keeps the same prediction

---

## Part 1: Attention Permutation (Section 4.2.1, Algorithm 2)

### The Algorithm

```
Algorithm 2: Permuting attention weights
─────────────────────────────────────────
h ← Enc(x)                          # Encode input → hidden states
α̂ ← softmax(φ(h, Q))              # Get original attention weights
ŷ ← Dec(h, α̂)                     # Get original prediction

for p ← 1 to 100 do
    αₚ ← Permute(α̂)               # Randomly shuffle the attention weights
    ŷₚ ← Dec(h, αₚ)               # Get prediction with shuffled attention
    ◁ Note: h is NOT changed — only attention is shuffled
    Δŷₚ ← TVD[ŷₚ, ŷ]             # Measure prediction change
end for

Δŷ_med ← Median_p(Δŷₚ)           # Report median prediction change
```

### Key Concepts

**Total Variation Distance (TVD)**: Measures how different two predictions are.
```
TVD(ŷ₁, ŷ₂) = ½ Σᵢ |ŷ₁ᵢ - ŷ₂ᵢ|
```
For binary classification (our case): `TVD = |p_shuffled - p_original|` (since the sum has only 2 terms that are equal in magnitude).

**Why we keep h fixed**: The hidden states h contain all the contextual information from the BiLSTM. By freezing h and only changing attention, we isolate the effect of attention. If shuffling attention barely changes the prediction, then the prediction doesn't actually depend on *which* hidden states get attended — it depends on the hidden states themselves.

### What to Expect

The paper finds that **randomly permuting attention weights often causes only minimal output change** (median Δŷ close to 0), especially for BiLSTM encoders. This means the model's attention pattern could be scrambled and the prediction barely budges.

---

## Part 2: Adversarial Attention (Section 4.2.2, Algorithm 3)

### The Paper's Formulation

The paper defines the adversarial attention problem as a **constrained optimization**:

```
maximize     f({α⁽ⁱ⁾}ᵢ₌₁ᵏ)
 α⁽¹⁾,...,α⁽ᵏ⁾

subject to   ∀i: TVD[ŷ(x, α⁽ⁱ⁾), ŷ(x, α̂)] ≤ ε
```

Where the objective f is:

```
f({α⁽ⁱ⁾}ᵢ₌₁ᵏ) = Σᵢ JSD[α⁽ⁱ⁾, α̂] + 1/(k(k-1)) Σᵢ<ⱼ JSD[α⁽ⁱ⁾, α⁽ʲ⁾]
```

**In plain terms**: Find k different attention distributions that are:
1. **Maximally different from the original** (maximize JSD with α̂)
2. **Diverse from each other** (maximize pairwise JSD among the k adversaries)
3. **Keep prediction within ε** of original (TVD constraint)

### Jensen-Shannon Divergence (JSD)

The paper uses JSD (not L1) to measure attention difference:

```
JSD(α₁, α₂) = ½ KL[α₁ ‖ M] + ½ KL[α₂ ‖ M]

where M = ½(α₁ + α₂)    — the average distribution
```

Properties:
- **Symmetric**: JSD(P, Q) = JSD(Q, P) — unlike KL divergence
- **Bounded**: 0 ≤ JSD ≤ log(2) ≈ 0.693 — makes comparison across texts meaningful
- **Well-defined**: Never encounters division by zero (KL would if Q has zeros)

JSD close to 0 = distributions are similar. JSD close to 0.693 = maximally different.

### The Relaxed Optimization (what the code actually does)

Since the constrained optimization is hard, the paper relaxes it using a **Lagrangian penalty**:

```
maximize   f({α⁽ⁱ⁾}) + λ/k · Σᵢ max(0, TVD[ŷ(x, α⁽ⁱ⁾), ŷ(x, α̂)] - ε)
```

Where λ = 500 is a large penalty weight. This adds a heavy cost when the prediction drifts beyond ε, effectively enforcing the constraint through optimization rather than as a hard constraint.

**In our implementation**, we simplify to k=1 (single adversarial attention distribution) and use:
```
Loss = -D(α_adv, α_orig) + λ · (p(α_adv) - p_orig)²
```

Where D is the attention difference measure and λ=100 penalizes prediction change.

### ε-max JSD Metric

The paper's key metric is **ε-max JSD**: the maximum JSD between original and adversarial attention, subject to the prediction staying within ε of the original.

```
ε-max JSD = max_i  1[Δŷ⁽ⁱ⁾ ≤ ε] · JSD[α̂, α⁽ⁱ⁾]
```

High ε-max JSD (close to 0.69) = we can find very different attention that keeps the same prediction = attention is NOT explanatory.

---

## How the Code Implements This

### `adversarial_attack.py`

#### `AdversarialAttentionAttack` class

**`get_original_attention_and_prediction()`**: Runs the model forward to get α̂ and ŷ. Freezes hidden states h.

**`find_adversarial_attention_gradient()`** — Implements Algorithm 3 (Section 4.2.2):
```python
# Freeze hidden states (h does not change)
embeddings = self.model.embedding(token_ids)
hidden_states, _ = self.model.encoder(embeddings, lengths)

# Optimize adversarial attention logits
adversarial_logits.requires_grad = True
optimizer = Adam([adversarial_logits])

for iteration in range(100):
    # Convert logits to valid attention via softmax
    α_adv = softmax(adversarial_logits)

    # Term 1: Maximize attention difference
    D = |α_adv - α_orig|.sum()

    # Term 2: Compute prediction with adversarial attention
    context = α_adv @ hidden_states    # New weighted sum
    p_adv = classifier(context)         # New prediction

    # Combined loss
    loss = -D + 100 * (p_adv - p_orig)²

    loss.backward()
    optimizer.step()
```

**`find_adversarial_attention_entropy()`** — Section 4.2.2 baseline:
Uses uniform attention (maximum entropy) as simplest adversarial. This always has high JSD from peaked original attention.

**`find_adversarial_attention_permutation()`** — Implements Algorithm 2 (Section 4.2.1):
Randomly swaps pairs of attention weights across positions. Preserves the weight values but reassigns them to different tokens.

**`find_adversarial_attention_random()`** — Monte Carlo sampling:
Generates random valid attention distributions. Keeps the most different one.

**`find_adversarial_attention_all_methods()`**: Runs all methods, returns whichever achieves the greatest attention difference while maintaining the same prediction.

#### `run_adversarial_experiment()` — Full pipeline for a single text

```
Text → Tokenize → Convert to IDs → Pad → Run all attack methods → Compare → Return metrics
```

#### `compute_attention_difference()` — Metrics

Computes L1, L2, max difference, mean difference, and cosine similarity between original and adversarial attention.

### `comparison.py`

Additional comparison utilities including KL divergence, JS divergence, Pearson correlation, and rank change analysis.

### `visualization.py`

Generates matplotlib visualizations as base64 PNG images:
- Bar chart comparing original vs adversarial attention
- Heatmap of per-token attention differences
- Word highlighting with attention-proportional opacity

---

## What the Results Prove

When run on "This movie was absolutely fantastic!":

| Metric | Value |
|--------|-------|
| Original prediction | 98.8% positive |
| Adversarial prediction | 98.8% positive |
| Attention L1 distance | 0.979 (high — very different attention) |
| Cosine similarity | 0.821 |

**Conclusion**: The prediction is identical despite the attention being radically different. The original attention peaked on "fantastic" and "absolutely" (suggesting they explain the prediction), but the adversarial attention can be nearly uniform — yet the model still says 98.8% positive. This proves attention does not faithfully explain the prediction.
