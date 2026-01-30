"""
Embedding layer module.
Step 3: Convert token indices to dense vector representations.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict


class EmbeddingLayer(nn.Module):
    """
    Embedding layer that converts token indices to dense vectors.
    
    Step 3: Embedding Lookup
    - Each token index is mapped to a d-dimensional vector
    - Supports pre-trained embeddings (GloVe) or learned embeddings
    
    Mathematical representation:
        x_i = E[t_i] ∈ ℝ^d
        
    Where:
        - E is the embedding matrix of shape (vocab_size, embedding_dim)
        - t_i is the token index
        - x_i is the resulting embedding vector
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        padding_idx: int = 0,
        dropout: float = 0.0,
        freeze: bool = False
    ):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of vocabulary (|V|)
            embedding_dim: Dimension of embeddings (d)
            padding_idx: Index of padding token (will have zero embedding)
            dropout: Dropout probability applied to embeddings
            freeze: If True, embeddings are not updated during training
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Create embedding layer
        # Shape: (vocab_size, embedding_dim)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Freeze embeddings if specified
        if freeze:
            self.embedding.weight.requires_grad = False
        
        # Initialize weights (Xavier uniform)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.xavier_uniform_(self.embedding.weight)
        # Ensure padding token has zero embedding
        with torch.no_grad():
            self.embedding.weight[self.padding_idx].fill_(0)
    
    def load_pretrained(
        self,
        vectors: np.ndarray,
        freeze: bool = True
    ):
        """
        Load pre-trained embedding vectors.
        
        Args:
            vectors: NumPy array of shape (vocab_size, embedding_dim)
            freeze: Whether to freeze embeddings after loading
        """
        assert vectors.shape == (self.vocab_size, self.embedding_dim), \
            f"Shape mismatch: expected {(self.vocab_size, self.embedding_dim)}, got {vectors.shape}"
        
        self.embedding.weight.data.copy_(torch.from_numpy(vectors))
        
        # Ensure padding token has zero embedding
        with torch.no_grad():
            self.embedding.weight[self.padding_idx].fill_(0)
        
        if freeze:
            self.embedding.weight.requires_grad = False
            print("Embeddings frozen (will not be updated during training)")
    
    def load_glove(
        self,
        glove_path: str,
        vocab: Dict[str, int],
        freeze: bool = True
    ):
        """
        Load GloVe embeddings for vocabulary.
        
        Args:
            glove_path: Path to GloVe file (e.g., glove.6B.300d.txt)
            vocab: Dictionary mapping tokens to indices
            freeze: Whether to freeze embeddings after loading
        """
        print(f"Loading GloVe embeddings from {glove_path}...")
        
        # Initialize with random embeddings
        pretrained = np.random.uniform(-0.25, 0.25, (self.vocab_size, self.embedding_dim))
        
        # Load GloVe vectors
        found = 0
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                    idx = vocab[word]
                    vector = np.array(parts[1:], dtype=np.float32)
                    if len(vector) == self.embedding_dim:
                        pretrained[idx] = vector
                        found += 1
        
        print(f"Found {found}/{len(vocab)} words in GloVe")
        
        # Load into embedding layer
        self.load_pretrained(pretrained, freeze=freeze)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert token indices to embeddings.
        
        Args:
            token_ids: Tensor of token indices
                      Shape: (batch_size, seq_length)
        
        Returns:
            Embedding vectors
            Shape: (batch_size, seq_length, embedding_dim)
        """
        # Lookup embeddings
        # (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embeddings = self.embedding(token_ids)
        
        # Apply dropout if specified
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def __repr__(self):
        return (
            f"EmbeddingLayer("
            f"vocab_size={self.vocab_size}, "
            f"embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx})"
        )
