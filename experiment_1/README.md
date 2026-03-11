# Experiment 1: Attention vs Feature Importance Correlation (Paper Section 4.1)

## What This Experiment Tests

**Question**: Do attention weights correlate with other measures of feature importance?

If attention truly explains why a model made a prediction, then words with high attention should also be the words that are most important according to **independent** importance measures:
- **Gradient-based importance**: How much does the output change when we slightly perturb each input?
- **Leave-One-Out (LOO) importance**: How much does the output change when we completely remove each word?

If attention is a good explanation, these should agree. If they don't correlate, attention is not reliably explaining what matters.

---

## The Algorithm (from the Paper)

The paper specifies this as **Algorithm 1**:

### Step 1: Get model outputs
```
h ← Enc(x)                         # Encode input to get hidden states
α̂ ← softmax(φ(h, Q))              # Get attention weights
ŷ ← Dec(h, α̂)                     # Get prediction
```

### Step 2: Compute gradient-based feature importance

For each token position t:
```
gₜ = |Σ_{w=1}^{|V|} 1[x_{tw} = 1] · ∂ŷ/∂x_{tw}|
```

**What this means in plain terms**: 
- Take the gradient of the output ŷ with respect to the one-hot input at position t
- Since the input is one-hot encoded, only one entry per position is non-zero
- The gradient tells us: "If I infinitesimally changed this input, how much would the output change?"
- Larger gradient magnitude = more important token

**In practice with embeddings**: Since we use embedding layers (not raw one-hot), we compute:
```python
embedding = model.embedding(token_ids)     # (batch, seq_len, embed_dim)
embedding.retain_grad()                     # Need gradients w.r.t. embeddings

prediction = model(token_ids)               # Forward pass
prediction.backward()                       # Backpropagate

# Gradient magnitude per token = L2 norm of gradient across embedding dims
gradient_importance = embedding.grad.norm(dim=-1)  # (batch, seq_len)
```

The L2 norm aggregates the gradient across all embedding dimensions into a single importance score per token.

**Why we disconnect the attention module**: The paper explicitly states that the gradient should NOT flow through the attention layer. This is because we want to measure "how important is this input token to the output, *given the current attention distribution*?" — not how changing the input would change the attention itself. We achieve this by detaching the attention weights from the computation graph.

### Step 3: Compute Leave-One-Out (LOO) importance

For each token position t:
```
Δŷₜ = TVD(ŷ(x₋ₜ), ŷ(x))
```

Where:
- `x₋ₜ` = the input with token at position t removed (replaced with padding)
- `TVD` = Total Variation Distance = ½ Σᵢ |ŷ₁ᵢ - ŷ₂ᵢ| (for our binary case: |ŷ(x₋ₜ) - ŷ(x)|)
- This directly measures: "If I **remove** this word entirely, how much does the prediction change?"

**In code**:
```python
for t in range(seq_length):
    masked_ids = token_ids.clone()
    masked_ids[0, t] = PAD_IDX          # Replace token with padding
    
    with torch.no_grad():
        masked_pred = model(masked_ids)  # Prediction without token t
    
    loo_importance[t] = abs(original_pred - masked_pred)  # TVD
```

LOO is the most intuitive importance measure: a word is important if removing it changes the prediction.

### Step 4: Compute Kendall's τ correlation

```
τ_g   = Kendall-τ(α̂, g)      # Correlation between attention and gradients
τ_loo = Kendall-τ(α̂, Δŷ)     # Correlation between attention and LOO scores
```

**Why Kendall's τ (not Pearson's r)?**

Kendall's τ measures **rank correlation** — do the rankings agree? It asks: "If token A has higher attention than token B, does token A also have higher gradient importance than token B?"

```
τ = (concordant_pairs - discordant_pairs) / total_pairs

Where:
- Concordant pair: αᵢ > αⱼ AND gᵢ > gⱼ  (rankings agree)
- Discordant pair: αᵢ > αⱼ AND gᵢ < gⱼ  (rankings disagree)
```

Range: [-1, 1]
- τ = 1: Perfect agreement in rankings
- τ = 0: No correlation (random)  
- τ = -1: Perfectly inverted rankings

**Why rank correlation?** Attention values and gradient magnitudes have very different scales. We don't care if attention weight 0.27 equals gradient magnitude 3.14 — we care if they RANK words the same way. Kendall's τ is robust to monotonic transformations of either measure.

---

## How the Code Implements This

### `experiment_1/feature_importance.py`

The `FeatureImportanceAnalyzer` class implements Algorithm 1:

1. **`compute_gradient_importance()`**: Forward pass with gradient tracking on embeddings, then `embedding.grad.norm(dim=-1)` per token
2. **`compute_loo_importance()`**: Loops over each token position, masks it out, re-runs the model, records prediction change
3. **`compute_correlations()`**: Computes Kendall τ between attention and each importance measure

### `experiment_1/main.py`

Loads the BiLSTM model (and optionally transformer), runs the analysis on sample texts, prints:
- Per-token attention weights vs gradient vs LOO scores
- Kendall τ_g and τ_loo correlations
- Histogram data for all test instances

---

## What the Results Should Show

**Paper's finding for IMDB (BiLSTM)**:
- τ_g ≈ 0.37 ± 0.08 (weak correlation with gradients)
- τ_loo ≈ 0.30 ± 0.07 (weak correlation with LOO)

This means attention and gradient/LOO rankings agree only ~35% of the time — barely better than random — proving attention is **not** reliably tracking what actually matters for the prediction.

**For comparison, with simple average encoder**: τ_g ≈ 0.65 (much stronger), showing the issue is specific to BiLSTM hidden states where attention operates on contextualized representations.

---

## Why This Matters

If attention correlated well with gradients/LOO, you could argue "attention shows what matters." But since it doesn't:

1. The model might attend to "fantastic" (high attention) BUT the prediction actually depends on hidden state patterns spread across ALL tokens (because BiLSTM contextualizes each position)
2. **Attention operates on hidden states, not raw inputs** — `h_t` at position t contains information from ALL words (due to bidirectional LSTM), so attending to position t ≠ attending to word t
3. This is the fundamental weakness: the attention mechanism's input (hidden states) already entangles all position information, making the attention-to-input mapping unreliable as an explanation
