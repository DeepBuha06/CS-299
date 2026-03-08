# Transformer Model Pipeline: Complete Explanation

## Table of Contents
1. [Overview](#overview)
2. [Step 1: Raw Text Input](#step-1-raw-text-input)
3. [Step 2: Tokenization](#step-2-tokenization)
4. [Step 3: Dataset Creation and Batching](#step-3-dataset-creation-and-batching)
5. [Step 4: Model Architecture](#step-4-model-architecture)
6. [Step 5: Forward Pass](#step-5-forward-pass)
7. [Step 6: Training Process](#step-6-training-process)
8. [Step 7: Evaluation and Inference](#step-7-evaluation-and-inference)
9. [Complete Data Flow Diagram](#complete-data-flow-diagram)
10. [Key Differences from BiLSTM](#key-differences-from-bilstm)

---

## Overview

This document explains the complete pipeline of the **Transformer-based Sentiment Classifier** built on top of **DistilBERT** (a distilled version of BERT). The model classifies IMDB movie reviews as **positive** or **negative** sentiment.

**Key Technologies:**
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Binary sentiment classification
- **Dataset**: IMDB movie reviews (25,000 training, 25,000 testing)
- **Framework**: PyTorch + HuggingFace Transformers

---

## Step 1: Raw Text Input

### 1.1 Source Data Structure
```
data/raw/imdb/
├── train/
│   ├── neg/  (12,500 negative reviews)
│   │   ├── 0_2.txt
│   │   ├── 1_3.txt
│   │   └── ...
│   └── pos/  (12,500 positive reviews)
│       ├── 0_9.txt
│       ├── 1_7.txt
│       └── ...
└── test/
    ├── neg/  (12,500 negative reviews)
    └── pos/  (12,500 positive reviews)
```

### 1.2 Example Raw Text
```
"This movie was absolutely fantastic! The acting was superb, 
the plot was engaging, and I couldn't take my eyes off the screen. 
Highly recommended for anyone who loves great cinema."
```

**Label**: `1` (positive)

---

## Step 2: Tokenization

Tokenization converts raw text into numeric tokens that the model can process. The transformer uses **WordPiece tokenization** via DistilBERT's tokenizer.

### 2.1 Loading the Tokenizer
```python
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# Vocabulary size: 30,522 tokens
```

### 2.2 Tokenization Process

**Step-by-step for example text:**

```python
text = "This movie was absolutely fantastic!"

# Tokenize
encoding = tokenizer(
    text,
    truncation=True,       # Truncate if longer than max_length
    max_length=256,        # Maximum sequence length
    padding="max_length",  # Pad to max_length
    return_tensors="pt"    # Return PyTorch tensors
)
```

**Output Structure:**
```python
{
    'input_ids': tensor([[  101,  2023,  3185,  2001,  7078,  8560,   999,   102,     0, ...]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]])
}
```

### 2.3 Understanding the Components

#### Input IDs
Token IDs from the vocabulary:
- **101** → `[CLS]` (Classification token - marks beginning)
- **2023** → "this"
- **3185** → "movie"
- **2001** → "was"
- **7078** → "absolutely"
- **8560** → "fantastic"
- **999** → "!"
- **102** → `[SEP]` (Separator token - marks end)
- **0** → `[PAD]` (Padding tokens to reach max_length=256)

#### Attention Mask
Binary mask indicating which tokens are real (1) vs padding (0):
```
[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ..., 0]
 ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
 CLS real tokens...  SEP  PAD tokens (ignored)
```

### 2.4 WordPiece Tokenization Details

For unknown or rare words, DistilBERT breaks them into subwords:
```
"unbelievable" → ["un", "##believable"]
"preprocessing" → ["pre", "##process", "##ing"]
```

This allows the model to handle out-of-vocabulary words!

---

## Step 3: Dataset Creation and Batching

### 3.1 Dataset Class (`IMDBTransformerDataset`)

The dataset class loads all reviews and provides them to the model:

```python
class IMDBTransformerDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512, split="train"):
        self.data_dir = Path(data_dir) / split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load all reviews from neg/ and pos/ folders
        self.reviews, self.labels = self._load_reviews()
```

### 3.2 Loading Process

**For each file:**
1. Read text from `.txt` file
2. Assign label: `0` for neg/, `1` for pos/
3. Store in lists

**Example:**
```python
reviews = [
    "This movie was terrible...",    # From neg/
    "Amazing film, loved it!",       # From pos/
    ...
]
labels = [0, 1, ...]  # Corresponding labels
```

### 3.3 Getting a Single Sample

When you access `dataset[0]`, it:
1. Gets the review text and label
2. Tokenizes the text
3. Returns a dictionary

```python
def __getitem__(self, idx):
    review = self.reviews[idx]
    label = self.labels[idx]
    
    encoding = self.tokenizer(
        review,
        truncation=True,
        max_length=self.max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    return {
        "input_ids": encoding["input_ids"].squeeze(0),      # Shape: (256,)
        "attention_mask": encoding["attention_mask"].squeeze(0),  # Shape: (256,)
        "label": torch.tensor(label, dtype=torch.long)      # Shape: ()
    }
```

### 3.4 Creating Batches with DataLoader

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=16,      # Process 16 reviews at once
    shuffle=True,       # Shuffle for better training
    collate_fn=collate_fn  # Stack samples into batches
)
```

**Collate Function** combines multiple samples:
```python
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    
    return {
        "input_ids": input_ids,        # Shape: (16, 256)
        "attention_mask": attention_mask,  # Shape: (16, 256)
        "labels": labels               # Shape: (16,)
    }
```

**Batch dimensions:**
- `input_ids`: **(batch_size=16, seq_length=256)**
- `attention_mask`: **(16, 256)**
- `labels`: **(16,)**

---

## Step 4: Model Architecture

The model consists of three main components:

```python
class TransformerClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        super().__init__()
        
        # 1. Pre-trained DistilBERT encoder
        self.distilbert = DistilBertModel.from_pretrained(
            model_name,
            output_attentions=True
        )
        
        # 2. Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 3. Linear classification head
        self.classifier = nn.Linear(768, num_labels)  # 768 is DistilBERT's hidden size
```

### 4.1 DistilBERT Architecture

**DistilBERT** is a smaller, faster version of BERT with:
- **6 Transformer layers** (BERT has 12)
- **12 attention heads** per layer
- **768-dimensional hidden states**
- **~66M trainable parameters**

**Each Transformer Layer contains:**
1. **Multi-Head Self-Attention** (12 heads)
2. **Layer Normalization**
3. **Feed-Forward Network** (768 → 3072 → 768)
4. **Layer Normalization**
5. **Residual connections** around each sub-layer

### 4.2 Parameter Count

```python
Total parameters: 66,955,010
Trainable parameters: 66,955,010
```

**Breakdown:**
- Token embeddings: 30,522 × 768 = 23,440,896
- Position embeddings: 512 × 768 = 393,216
- 6 Transformer layers: ~40M parameters
- Classification head: 768 × 2 = 1,536

---

## Step 5: Forward Pass

This is where the magic happens! Let's trace a single review through the model.

### 5.1 Input to the Model

```python
# Batch from DataLoader
input_ids = tensor([[101, 2023, 3185, ...]])       # (16, 256)
attention_mask = tensor([[1, 1, 1, ...]])          # (16, 256)

# Move to GPU if available
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
```

### 5.2 Step-by-Step Forward Pass

```python
def forward(self, input_ids, attention_mask, return_attention=False):
    # 1. Pass through DistilBERT
    outputs = self.distilbert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=return_attention
    )
```

#### 5.2.1 Inside DistilBERT: Token Embeddings

**Input IDs → Embeddings**
```python
# Token embeddings (converts IDs to vectors)
token_embeddings = embedding_layer(input_ids)  # (16, 256, 768)

# Position embeddings (encodes position in sequence)
position_ids = [0, 1, 2, 3, ..., 255]
position_embeddings = position_embedding(position_ids)  # (1, 256, 768)

# Combine
embeddings = token_embeddings + position_embeddings  # (16, 256, 768)
embeddings = layer_norm(embeddings)
```

**Shape**: **(batch=16, seq_length=256, hidden_size=768)**

Each token is now a 768-dimensional vector!

#### 5.2.2 Transformer Layer 1-6: Self-Attention

**For EACH of the 6 layers:**

**A. Multi-Head Self-Attention**
```python
# Input: (16, 256, 768)

# Project to Query, Key, Value for 12 heads
Q = linear_q(x)  # (16, 256, 768)
K = linear_k(x)
V = linear_v(x)

# Reshape for multi-head: (16, 12, 256, 64)
# 12 heads, each with dimension 768/12 = 64
Q = Q.view(16, 256, 12, 64).transpose(1, 2)
K = K.view(16, 256, 12, 64).transpose(1, 2)
V = V.view(16, 256, 12, 64).transpose(1, 2)

# Compute attention scores
scores = Q @ K.transpose(-2, -1) / sqrt(64)  # (16, 12, 256, 256)
# Each token attends to every other token!

# Apply attention mask (ignore padding)
scores = scores.masked_fill(attention_mask == 0, -1e9)

# Softmax to get attention weights
attention_weights = softmax(scores, dim=-1)  # (16, 12, 256, 256)

# Apply attention to values
attention_output = attention_weights @ V  # (16, 12, 256, 64)

# Concatenate heads
attention_output = attention_output.transpose(1, 2).contiguous()
attention_output = attention_output.view(16, 256, 768)
```

**Attention Interpretation:**
- For position 0 ([CLS] token), attention_weights[0, :, 0, :] shows how much it attends to all other tokens
- High attention = that token is important for understanding [CLS]

**B. Add & Norm (Residual Connection)**
```python
x = layer_norm(x + attention_output)  # (16, 256, 768)
```

**C. Feed-Forward Network**
```python
# Two-layer MLP with GELU activation
ffn_output = linear2(gelu(linear1(x)))  # 768 → 3072 → 768
```

**D. Add & Norm Again**
```python
x = layer_norm(x + ffn_output)  # (16, 256, 768)
```

**Repeat this for all 6 layers!**

#### 5.2.3 Extract [CLS] Token Representation

After all 6 transformer layers:
```python
# outputs.last_hidden_state: (16, 256, 768)
# All token representations from final layer

# Extract [CLS] token (first token, position 0)
cls_output = outputs.last_hidden_state[:, 0, :]  # (16, 768)
```

**Why [CLS]?**
The [CLS] token at position 0 is a special token trained to aggregate information from the entire sequence through self-attention. It serves as the **sequence-level representation**.

#### 5.2.4 Classification Head

```python
# Apply dropout
pooled_output = self.dropout(cls_output)  # (16, 768)

# Linear layer to get logits
logits = self.classifier(pooled_output)  # (16, 2)
```

**Logits shape**: **(16, 2)**
- 16 samples in batch
- 2 classes: [negative, positive]

**Example logits:**
```python
logits = [[-2.3, 3.1],   # Strong positive prediction
          [1.8, -1.2],   # Strong negative prediction
          ...]
```

#### 5.2.5 Convert to Probabilities

```python
probabilities = torch.softmax(logits, dim=-1)  # (16, 2)
# [[0.01, 0.99],   # 99% positive
#  [0.94, 0.06],   # 94% negative
#  ...]
```

### 5.3 Complete Forward Pass Summary

```
Input Text: "This movie was fantastic!"
     ↓
Tokenization: [CLS] this movie was fantastic ! [SEP] [PAD] ...
     ↓
Token IDs: [101, 2023, 3185, 2001, 8560, 999, 102, 0, ...]
     ↓
Embeddings: (1, 256, 768)
     ↓
Transformer Layer 1: Self-Attention + FFN → (1, 256, 768)
     ↓
Transformer Layer 2: Self-Attention + FFN → (1, 256, 768)
     ↓
...
     ↓
Transformer Layer 6: Self-Attention + FFN → (1, 256, 768)
     ↓
Extract [CLS]: (1, 768)
     ↓
Dropout: (1, 768)
     ↓
Linear Classifier: (1, 2)
     ↓
Logits: [-2.3, 3.1]
     ↓
Softmax: [0.01, 0.99]
     ↓
Prediction: Positive (99% confidence)
```

---

## Step 6: Training Process

### 6.1 Training Configuration

```python
# Hyperparameters
LEARNING_RATE = 2e-5          # Small LR for fine-tuning
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2      # Effective batch size = 32
NUM_EPOCHS = 1                 # Single epoch
WARMUP_RATIO = 0.1            # Warmup for stability
MAX_GRAD_NORM = 1.0           # Gradient clipping
```

### 6.2 Optimizer and Scheduler

```python
# AdamW optimizer (Adam with weight decay)
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01
)

# Linear warmup + decay scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

**Learning rate schedule:**
```
LR
^
|     ___________________ (plateau at 2e-5)
|    /                    \
|   /                      \
|  /                        \_____ (decay to 0)
| /
|/
+---------------------------------> Steps
0   warmup              training end
```

### 6.3 Training Loop (One Epoch)

```python
for epoch in range(NUM_EPOCHS):
    model.train()  # Set to training mode
    
    for step, batch in enumerate(train_loader):
        # 1. Get batch data
        input_ids = batch["input_ids"].to(device)        # (16, 256)
        attention_mask = batch["attention_mask"].to(device)  # (16, 256)
        labels = batch["labels"].to(device)              # (16,)
        
        # 2. Forward pass
        logits, _ = model(input_ids, attention_mask)     # (16, 2)
        
        # 3. Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        # 4. Backward pass
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()  # Compute gradients
        
        # 5. Update weights (every 2 steps)
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            optimizer.step()   # Update parameters
            scheduler.step()   # Update learning rate
            optimizer.zero_grad()  # Reset gradients
```

### 6.4 Loss Function: Cross-Entropy

Cross-entropy measures how different the predictions are from the true labels.

**Formula:**
```
Loss = -log(probability of correct class)
```

**Example:**
```python
# True label: 1 (positive)
# Predicted probabilities: [0.1, 0.9]
loss = -log(0.9) = 0.105  # Low loss (good prediction)

# True label: 1 (positive)
# Predicted probabilities: [0.7, 0.3]
loss = -log(0.3) = 1.204  # High loss (bad prediction)
```

For a batch:
```python
logits = [[-2.3, 3.1], [1.8, -1.2], ...]  # (16, 2)
labels = [1, 0, ...]                       # (16,)

loss = CrossEntropyLoss(logits, labels)
# Averages loss across all 16 samples
```

### 6.5 Gradient Descent Update

**Backpropagation:**
```python
loss.backward()  # Computes ∂loss/∂weight for all parameters
```

This calculates gradients flowing backward through:
1. Cross-entropy loss
2. Linear classifier
3. Dropout
4. [CLS] token
5. Transformer layer 6
6. ...
7. Transformer layer 1
8. Embeddings

**Parameter update:**
```python
# AdamW optimizer updates each parameter:
weight_new = weight_old - learning_rate × gradient + weight_decay_term
```

### 6.6 Training Progress

**Typical output:**
```
Epoch 1/1
Training: 100%|████████| 1563/1563 [42:15<00:00]
  loss=0.2134, acc=0.9234
  
Test Loss: 0.2567, Accuracy: 0.8956
Precision: 0.8912, Recall: 0.9001, F1: 0.8956
```

### 6.7 Saving the Model

```python
checkpoint = {
    "model_state_dict": model.state_dict(),  # All weights
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "accuracy": best_accuracy
}
torch.save(checkpoint, "checkpoints/transformer_model.pt")
```

---

## Step 7: Evaluation and Inference

### 7.1 Loading a Trained Model

```python
model = TransformerClassifier(
    model_name="distilbert-base-uncased",
    num_labels=2
)
checkpoint = torch.load("checkpoints/transformer_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()  # Set to evaluation mode (disables dropout)
```

### 7.2 Predicting on New Text

```python
def predict_sentiment(text):
    # 1. Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # 2. Forward pass (no gradient computation)
    with torch.no_grad():
        logits, attention = model(
            input_ids,
            attention_mask,
            return_attention=True
        )
        probabilities = torch.softmax(logits, dim=-1)
    
    # 3. Get prediction
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class].item()
    
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "prob_positive": probabilities[0, 1].item(),
        "prob_negative": probabilities[0, 0].item()
    }
```

**Example:**
```python
text = "This movie was absolutely amazing! Best film I've seen this year."
result = predict_sentiment(text)

# Output:
# {
#     "sentiment": "Positive",
#     "confidence": 0.9876,
#     "prob_positive": 0.9876,
#     "prob_negative": 0.0124
# }
```

### 7.3 Attention Visualization

The model can return attention weights showing which words it focuses on:

```python
def visualize_attention(text):
    # Get tokens and attention
    tokens = tokenizer.tokenize(text)
    _, attention = model(input_ids, attention_mask, return_attention=True)
    
    # attention: (1, 256) - attention from [CLS] to each token
    attention = attention[0, :len(tokens)].cpu().numpy()
    
    # Plot
    plt.figure(figsize=(12, 3))
    plt.bar(range(len(tokens)), attention)
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.ylabel("Attention Weight")
    plt.title("Attention from [CLS] Token")
```

**Example visualization:**
```
Attention Weight
^
1.0|
   |     ██
0.8|     ██        ██
   |     ██  ██    ██
0.6| ██  ██  ██    ██
   | ██  ██  ██ ██ ██  ██
0.4| ██  ██  ██ ██ ██  ██
   +---------------------------
    [CLS] movie was fantastic! [SEP]
```

Words with higher bars (like "fantastic") are more important for the prediction!

### 7.4 Evaluation Metrics

```python
# On test set (25,000 reviews)
test_metrics = {
    "accuracy": 0.8956,     # 89.56% correct predictions
    "precision": 0.8912,    # Of predicted positive, 89.12% are truly positive
    "recall": 0.9001,       # Of actual positive, 90.01% are detected
    "f1": 0.8956           # Harmonic mean of precision and recall
}
```

**Confusion Matrix:**
```
                Predicted
                Neg    Pos
    Actual Neg  11,234  1,266
           Pos  1,234  11,266
```

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. RAW TEXT INPUT                                               │
└─────────────────────────────────────────────────────────────────┘
    "This movie was absolutely fantastic!"
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. FILE LOADING (IMDBTransformerDataset)                       │
│    - Read from data/raw/imdb/train/pos/*.txt                   │
│    - Assign labels: neg=0, pos=1                               │
└─────────────────────────────────────────────────────────────────┘
    reviews = ["This movie was..."], labels = [1]
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. TOKENIZATION (DistilBertTokenizer)                          │
│    - Text → Token IDs using WordPiece                         │
│    - Add special tokens: [CLS], [SEP]                          │
│    - Pad/truncate to max_length=256                            │
│    - Create attention mask                                      │
└─────────────────────────────────────────────────────────────────┘
    input_ids: [101, 2023, 3185, 2001, 7078, 8560, 999, 102, 0, ...]
    attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 0, ...]
    Shape: (256,)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. BATCHING (DataLoader + collate_fn)                          │
│    - Stack samples into batches                                 │
│    - batch_size = 16                                           │
└─────────────────────────────────────────────────────────────────┘
    input_ids: (16, 256)
    attention_mask: (16, 256)
    labels: (16,)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. EMBEDDING LAYER                                              │
│    - Token IDs → token embeddings (768-dim)                    │
│    - Add position embeddings                                    │
│    - Layer normalization                                        │
└─────────────────────────────────────────────────────────────────┘
    embeddings: (16, 256, 768)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. TRANSFORMER LAYER 1                                          │
│    ┌─────────────────────────────────────┐                    │
│    │ Multi-Head Self-Attention (12 heads)│                    │
│    │   - Query, Key, Value projections   │                    │
│    │   - Attention scores & weights      │                    │
│    │   - Weighted sum of values          │                    │
│    └─────────────────────────────────────┘                    │
│                   ↓                                             │
│    ┌─────────────────────────────────────┐                    │
│    │ Add & Normalize (Residual)          │                    │
│    └─────────────────────────────────────┘                    │
│                   ↓                                             │
│    ┌─────────────────────────────────────┐                    │
│    │ Feed-Forward Network                │                    │
│    │   768 → 3072 → 768                  │                    │
│    └─────────────────────────────────────┘                    │
│                   ↓                                             │
│    ┌─────────────────────────────────────┐                    │
│    │ Add & Normalize (Residual)          │                    │
│    └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
    hidden_states: (16, 256, 768)
                    ↓
    [Repeat for Transformer Layers 2-6]
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. EXTRACT [CLS] TOKEN                                          │
│    - Take first token from final layer                         │
│    - Aggregates full sequence information                      │
└─────────────────────────────────────────────────────────────────┘
    cls_output: (16, 768)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. DROPOUT                                                      │
│    - Randomly drop 10% of activations (training only)          │
└─────────────────────────────────────────────────────────────────┘
    pooled_output: (16, 768)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 9. CLASSIFICATION HEAD (Linear Layer)                          │
│    - Linear: 768 → 2                                           │
└─────────────────────────────────────────────────────────────────┘
    logits: (16, 2)
    Example: [[-2.3, 3.1], [1.8, -1.2], ...]
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 10. SOFTMAX (Inference) / LOSS (Training)                      │
│                                                                 │
│ INFERENCE:                      TRAINING:                      │
│   probabilities = softmax(logits) loss = CrossEntropy(logits,  │
│   [0.01, 0.99] → 99% positive      labels)                     │
│                                     ↓                           │
│                                   loss.backward()               │
│                                     ↓                           │
│                                   optimizer.step()              │
└─────────────────────────────────────────────────────────────────┘
    probabilities: (16, 2)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 11. PREDICTION                                                  │
│    - argmax(probabilities)                                     │
│    - 0 = Negative, 1 = Positive                                │
└─────────────────────────────────────────────────────────────────┘
    predictions: [1, 0, 1, 1, 0, ...]
    
    FINAL OUTPUT: "Positive (99% confidence)"
```

---

## Key Differences from BiLSTM

| Aspect | BiLSTM | Transformer |
|--------|--------|-------------|
| **Tokenization** | Custom vocabulary | WordPiece (30,522 tokens) |
| **Unknown words** | `<UNK>` token | Subword splitting |
| **Sequence processing** | Sequential (left→right, right→left) | Parallel (all at once) |
| **Position encoding** | Implicit in LSTM | Explicit position embeddings |
| **Attention** | None (hidden states only) | Multi-head self-attention (12 heads × 6 layers) |
| **Context** | Limited by hidden size | Global context via attention |
| **Training time** | Faster (fewer parameters) | Slower (66M parameters) |
| **Accuracy** | ~86% (custom trained) | ~90% (pre-trained + fine-tuned) |
| **Parameters** | ~5M | ~66M |
| **Memory usage** | Lower | Higher |
| **Pre-training** | None | Pre-trained on huge corpus |

---

## Summary

The transformer pipeline transforms raw text into sentiment predictions through these key stages:

1. **Text Loading**: Read review files from disk
2. **Tokenization**: Convert text to numeric tokens using WordPiece
3. **Batching**: Group samples for efficient processing
4. **Embedding**: Convert tokens to dense vectors with position info
5. **Transformer Layers**: 6 layers of self-attention + feed-forward networks
6. **Pooling**: Extract [CLS] token as sequence representation
7. **Classification**: Linear layer produces logits for 2 classes
8. **Training**: Optimize with cross-entropy loss and AdamW
9. **Inference**: Softmax converts logits to probabilities

**Key advantages:**
- ✅ **Pre-trained knowledge**: Leverages DistilBERT's understanding of language
- ✅ **Global context**: Self-attention sees entire sequence
- ✅ **Subword tokenization**: Handles unknown words gracefully
- ✅ **Parallel processing**: Fast inference (not sequential like LSTM)
- ✅ **Interpretable**: Attention weights show what the model focuses on

This architecture represents the **state-of-the-art** approach for text classification tasks!
