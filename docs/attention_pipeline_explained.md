# Attention-Based Text Classification Pipeline

> **Reference**: Section 2 (Preliminaries and Assumptions) from *"Attention is Not Explanation"* — Jain & Wallace (2019)

This document provides a detailed walkthrough of the 10-step pipeline for attention-based text classification, using IMDB sentiment analysis as the running example.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION-BASED CLASSIFICATION                       │
├─────────────────────────────────────────────────────────────────────────┤
│  Input Text → Tokens → Embeddings → Encoder → Attention → Prediction   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Raw Input

### What happens?
You start with a raw text review from the IMDB dataset.

### Example
```
"This movie was absolutely fantastic! The acting was superb and the 
storyline kept me engaged throughout."
```

### Details
- **IMDB Dataset**: Contains 50,000 movie reviews (25k train, 25k test)
- **Task**: Binary classification → Positive or Negative sentiment
- **Label**: This review would be labeled as **Positive (1)**

### Mathematical Representation
$$
\text{Input} = \text{Raw String } S
$$

---

## Step 2: Tokenization

### What happens?
The raw text is split into individual tokens (words, subwords, or characters).

### Example
```python
Input:  "This movie was absolutely fantastic!"

Output: ["This", "movie", "was", "absolutely", "fantastic", "!"]
        [  t₁  ,   t₂   ,  t₃  ,     t₄     ,     t₅     , t₆ ]
```

### Details
- **Tokenizer choices**: 
  - Word-level: Split on whitespace and punctuation
  - Subword: BPE, WordPiece, SentencePiece
  - Character-level: Each character is a token

- **Vocabulary**: A predefined set of known tokens (typically 10k-50k words)
- **Special tokens**: 
  - `<PAD>` - Padding for batch processing
  - `<UNK>` - Unknown/out-of-vocabulary words
  - `<SOS>` / `<EOS>` - Start/End of sequence

### Mathematical Representation
$$
S \xrightarrow{\text{tokenize}} [t_1, t_2, t_3, ..., t_n]
$$

where $n$ = sequence length (number of tokens)

---

## Step 3: Embedding Lookup

### What happens?
Each token is converted into a dense vector representation using an embedding matrix.

### Example
```python
Token:     "fantastic"
Token ID:   4521
Embedding:  [0.23, -0.45, 0.87, ..., 0.12]  # d-dimensional vector
```

### Details
- **Embedding Matrix**: $E \in \mathbb{R}^{|V| \times d}$
  - $|V|$ = vocabulary size (e.g., 30,000)
  - $d$ = embedding dimension (e.g., 300)

- **Pre-trained Embeddings**:
  - **GloVe**: Trained on Wikipedia + Gigaword
  - **Word2Vec**: Skip-gram or CBOW
  - **FastText**: Handles subword information

- **Learned Embeddings**: Randomly initialized and trained end-to-end

### Mathematical Representation
$$
x_i = E[t_i] \in \mathbb{R}^d
$$

For the full sequence:
$$
X = [x_1, x_2, ..., x_n] \in \mathbb{R}^{n \times d}
$$

### Visual
```
Token IDs:    [  23  ,  156  ,   42  ,  891  ,  4521 ]
                 ↓       ↓       ↓       ↓       ↓
Embedding    ┌──────────────────────────────────────┐
Matrix E     │  ································    │ → x₁ = E[23]
(|V| × d)    │  ································    │ → x₂ = E[156]
             │  ································    │ → ...
             └──────────────────────────────────────┘
                 
Output:      [ x₁  ,  x₂  ,  x₃  ,  x₄  ,  x₅  ]
             (each xᵢ is a d-dimensional vector)
```

---

## Step 4: Encoder (Bidirectional LSTM)

### What happens?
The sequence of embeddings is passed through a recurrent neural network (typically BiLSTM) to capture contextual information.

### Example
```
Input embeddings:  [x₁, x₂, x₃, x₄, x₅]
                      ↓
                  [BiLSTM]
                      ↓
Hidden states:     [h₁, h₂, h₃, h₄, h₅]
```

### Details

#### Why BiLSTM?
- **Unidirectional LSTM**: Only captures left context (past)
- **Bidirectional LSTM**: Captures both left AND right context

#### Architecture
```
Forward LSTM:   →  →  →  →  →
                h₁ʲ h₂ʲ h₃ʲ h₄ʲ h₅ʲ

Backward LSTM:  ←  ←  ←  ←  ←
                h₁ᵇ h₂ᵇ h₃ᵇ h₄ᵇ h₅ᵇ

Combined:       hᵢ = [hᵢʲ ; hᵢᵇ]  (concatenation)
```

#### LSTM Cell Equations
For each timestep $t$:

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) & \text{(forget gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) & \text{(input gate)} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) & \text{(candidate)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t & \text{(cell state)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) & \text{(output gate)} \\
h_t &= o_t \odot \tanh(C_t) & \text{(hidden state)}
\end{aligned}
$$

### Mathematical Representation
$$
\vec{h}_i = \overrightarrow{\text{LSTM}}(x_i, \vec{h}_{i-1})
$$
$$
\overleftarrow{h}_i = \overleftarrow{\text{LSTM}}(x_i, \overleftarrow{h}_{i+1})
$$
$$
h_i = [\vec{h}_i ; \overleftarrow{h}_i] \in \mathbb{R}^{2 \cdot h_{dim}}
$$

Full sequence of hidden states:
$$
H = [h_1, h_2, ..., h_n] \in \mathbb{R}^{n \times 2h_{dim}}
$$

---

## Step 5: Attention Score Computation

### What happens?
For each hidden state, compute a scalar "attention score" that indicates its importance.

### Example
```
Hidden states:     [h₁,   h₂,   h₃,   h₄,   h₅]
                     ↓     ↓     ↓     ↓     ↓
                  [Attention Mechanism]
                     ↓     ↓     ↓     ↓     ↓
Attention scores:  [e₁,   e₂,   e₃,   e₄,   e₅]
                   [2.1,  0.5,  1.8,  3.2,  2.9]
```

### Details

#### Types of Attention in the Paper

**1. Additive Attention (Bahdanau)**
$$
e_i = v^T \cdot \tanh(W_h \cdot h_i + b)
$$
- $W_h \in \mathbb{R}^{d_a \times 2h_{dim}}$ - Weight matrix
- $v \in \mathbb{R}^{d_a}$ - Learnable vector
- $d_a$ - Attention hidden dimension

**2. Dot-Product Attention**
$$
e_i = q^T \cdot h_i
$$
- $q$ - Query vector (can be learned or computed)

**3. Scaled Dot-Product (Transformer-style)**
$$
e_i = \frac{q^T \cdot h_i}{\sqrt{d_k}}
$$

### Visual Representation
```
        h₁        h₂        h₃        h₄        h₅
        │         │         │         │         │
        ▼         ▼         ▼         ▼         ▼
    ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
    │ tanh  │ │ tanh  │ │ tanh  │ │ tanh  │ │ tanh  │
    │(Wh+b) │ │(Wh+b) │ │(Wh+b) │ │(Wh+b) │ │(Wh+b) │
    └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
        │         │         │         │         │
        ▼         ▼         ▼         ▼         ▼
    ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
    │  vᵀ·  │ │  vᵀ·  │ │  vᵀ·  │ │  vᵀ·  │ │  vᵀ·  │
    └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
        │         │         │         │         │
        ▼         ▼         ▼         ▼         ▼
       e₁        e₂        e₃        e₄        e₅
      (2.1)     (0.5)     (1.8)     (3.2)     (2.9)
```

---

## Step 6: Softmax Normalization

### What happens?
Convert raw attention scores into a probability distribution using softmax.

### Example
```
Raw scores:        [2.1,   0.5,   1.8,   3.2,   2.9]
                      ↓
                  [Softmax]
                      ↓
Attention weights: [0.12,  0.02,  0.09,  0.42,  0.35]
                   (sum = 1.0)
```

### Details

#### Softmax Function
$$
\alpha_i = \frac{\exp(e_i)}{\sum_{j=1}^{n} \exp(e_j)}
$$

#### Properties
- All weights are positive: $\alpha_i > 0$
- Weights sum to 1: $\sum_{i=1}^{n} \alpha_i = 1$
- Forms a valid probability distribution
- Higher scores → higher weights (exponential amplification)

#### Numerical Stability Trick
To avoid overflow with large values:
$$
\alpha_i = \frac{\exp(e_i - \max(e))}{\sum_{j=1}^{n} \exp(e_j - \max(e))}
$$

### Mathematical Representation
$$
\boldsymbol{\alpha} = \text{softmax}(\mathbf{e}) = [\alpha_1, \alpha_2, ..., \alpha_n]
$$

where $\alpha_i \geq 0$ and $\sum_i \alpha_i = 1$

### Visual
```
Raw Scores:    2.1     0.5     1.8     3.2     2.9
                │       │       │       │       │
                ▼       ▼       ▼       ▼       ▼
              ┌─────────────────────────────────────┐
              │           S O F T M A X             │
              │    αᵢ = exp(eᵢ) / Σ exp(eⱼ)         │
              └─────────────────────────────────────┘
                │       │       │       │       │
                ▼       ▼       ▼       ▼       ▼
Weights:      0.12    0.02    0.09    0.42    0.35
              ════════════════════════════════════
                        Sum = 1.0
```

---

## Step 7: Context Vector Computation

### What happens?
Compute a weighted sum of all hidden states using the attention weights.

### Example
```
Attention weights:  [0.12,  0.02,  0.09,  0.42,  0.35]
Hidden states:      [h₁,    h₂,    h₃,    h₄,    h₅]

Context vector:  c = 0.12·h₁ + 0.02·h₂ + 0.09·h₃ + 0.42·h₄ + 0.35·h₅
```

### Details

#### Weighted Sum
The context vector aggregates information from all positions, weighted by their "importance":

$$
c = \sum_{i=1}^{n} \alpha_i \cdot h_i
$$

#### Interpretation
- Tokens with **high attention weights** contribute more to the context vector
- Tokens with **low attention weights** are "ignored" (contribute minimally)
- The context vector is a **fixed-size representation** of the entire input

#### Dimension
$$
c \in \mathbb{R}^{2h_{dim}}
$$
(Same dimension as each hidden state $h_i$)

### Visual
```
Weights:     α₁=0.12   α₂=0.02   α₃=0.09   α₄=0.42   α₅=0.35
              │         │         │         │         │
              ▼         ▼         ▼         ▼         ▼
Hidden:      h₁        h₂        h₃        h₄        h₅
              │         │         │         │         │
              ▼         ▼         ▼         ▼         ▼
Weighted:  0.12·h₁   0.02·h₂   0.09·h₃   0.42·h₄   0.35·h₅
              │         │         │         │         │
              └─────────┴─────────┼─────────┴─────────┘
                                  ▼
                              Σ (sum)
                                  │
                                  ▼
                          Context Vector c
                        (fixed-size vector)
```

---

## Step 8: Classification Layer (Dense/Fully Connected)

### What happens?
The context vector is passed through one or more fully connected layers to produce logits.

### Example
```
Context vector c:  [0.23, -0.45, 0.87, ..., 0.12]  (size: 2h_dim)
                              ↓
                    [Fully Connected Layer]
                              ↓
Logits z:          [2.34]  (for binary) or [0.5, 2.1] (for multi-class)
```

### Details

#### Linear Transformation
$$
z = W_c \cdot c + b_c
$$

- $W_c \in \mathbb{R}^{k \times 2h_{dim}}$ - Weight matrix
- $b_c \in \mathbb{R}^{k}$ - Bias vector
- $k$ = number of classes (1 for binary, n for multi-class)

#### Multi-Layer Option
Sometimes multiple layers with non-linearity:
$$
h_{fc} = \text{ReLU}(W_1 \cdot c + b_1)
$$
$$
z = W_2 \cdot h_{fc} + b_2
$$

#### Dropout
Often applied before the classification layer:
$$
z = W_c \cdot \text{Dropout}(c, p) + b_c
$$
where $p$ is dropout probability (e.g., 0.5)

### Visual
```
Context Vector c
(dimension: 2h_dim = 512)
        │
        ▼
┌───────────────────┐
│  Dropout (0.5)    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Dense Layer     │
│  W·c + b          │
│  (512 → 1)        │
└─────────┬─────────┘
          │
          ▼
      Logit z
    (scalar: 2.34)
```

---

## Step 9: Output Probability (Sigmoid/Softmax)

### What happens?
Convert the raw logits into probabilities using an activation function.

### Example
```
Binary Classification:
Logit z = 2.34  →  sigmoid(2.34) = 0.912  →  P(Positive) = 91.2%

Multi-class Classification:
Logits z = [0.5, 2.1, 0.3]  →  softmax(z) = [0.16, 0.76, 0.08]
```

### Details

#### Binary Classification: Sigmoid
$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

- Output range: $(0, 1)$
- Interpretation: $\hat{y}$ = probability of positive class

#### Multi-class Classification: Softmax
$$
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}
$$

- Output: probability distribution over $k$ classes
- Sum of all probabilities = 1

### Visual (Binary)
```
Logit z = 2.34
     │
     ▼
┌─────────────────┐
│    SIGMOID      │
│  σ(z) = 1/(1+e⁻ᶻ)│
└────────┬────────┘
         │
         ▼
  Probability = 0.912
  
  P(Positive) = 91.2%
  P(Negative) = 8.8%
```

---

## Step 10: Final Prediction

### What happens?
Apply a decision threshold to convert probability into a discrete class label.

### Example
```
Probability:  P(Positive) = 0.912

Threshold:    0.5

Decision:     0.912 > 0.5  →  Predict: POSITIVE ✓
```

### Details

#### Binary Classification Threshold
$$
\hat{class} = 
\begin{cases} 
1 \text{ (Positive)} & \text{if } \hat{y} \geq \theta \\
0 \text{ (Negative)} & \text{if } \hat{y} < \theta
\end{cases}
$$

Default threshold $\theta = 0.5$, but can be tuned based on:
- Precision-Recall trade-off
- Business requirements
- Class imbalance

#### Multi-class Prediction
$$
\hat{class} = \arg\max_i(\hat{y}_i)
$$

Simply pick the class with highest probability.

### Loss Function (Training)

#### Binary Cross-Entropy
$$
\mathcal{L} = -[y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})]
$$

#### Categorical Cross-Entropy (Multi-class)
$$
\mathcal{L} = -\sum_{i=1}^{k} y_i \cdot \log(\hat{y}_i)
$$

### Final Output
```
┌─────────────────────────────────────────────┐
│                                             │
│   INPUT:  "This movie was absolutely        │
│            fantastic! The acting was        │
│            superb."                         │
│                                             │
│   OUTPUT: POSITIVE  (Confidence: 91.2%)     │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Complete Pipeline Summary

```
┌──────────────────────────────────────────────────────────────────────┐
│                         COMPLETE PIPELINE                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 1:  Raw Review Text                                           │
│              ↓                                                       │
│  STEP 2:  Tokenization → [t₁, t₂, ..., tₙ]                          │
│              ↓                                                       │
│  STEP 3:  Embedding Lookup → [x₁, x₂, ..., xₙ]                      │
│              ↓                                                       │
│  STEP 4:  BiLSTM Encoder → [h₁, h₂, ..., hₙ]                        │
│              ↓                                                       │
│  STEP 5:  Attention Scores → [e₁, e₂, ..., eₙ]                      │
│              ↓                                                       │
│  STEP 6:  Softmax → [α₁, α₂, ..., αₙ]  (Σαᵢ = 1)                    │
│              ↓                                                       │
│  STEP 7:  Context Vector → c = Σ αᵢhᵢ                               │
│              ↓                                                       │
│  STEP 8:  Dense Layer → z = Wc + b                                  │
│              ↓                                                       │
│  STEP 9:  Sigmoid → ŷ = σ(z)                                        │
│              ↓                                                       │
│  STEP 10: Threshold → POSITIVE / NEGATIVE                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Question from the Paper 🤔

> **"Do the attention weights (α) in Steps 5-7 actually EXPLAIN why the model made its prediction?"**

The paper argues: **Often NO!**

They show that:
1. Attention weights don't always correlate with gradient-based importance
2. You can often find **different attention distributions** that give the **same prediction**
3. Therefore, attention should not be treated as a faithful explanation

---

## References

- Jain, S., & Wallace, B. C. (2019). *Attention is not Explanation*. NAACL-HLT 2019.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
