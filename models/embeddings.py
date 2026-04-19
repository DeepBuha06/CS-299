import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict


class EmbeddingLayer(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        padding_idx: int = 0,
        dropout: float = 0.0,
        freeze: bool = False
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        if freeze:
            self.embedding.weight.requires_grad = False
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[self.padding_idx].fill_(0)
    
    def load_pretrained(
        self,
        vectors: np.ndarray,
        freeze: bool = True
    ):
        assert vectors.shape == (self.vocab_size, self.embedding_dim), \
            f"Shape mismatch: expected {(self.vocab_size, self.embedding_dim)}, got {vectors.shape}"
        
        self.embedding.weight.data.copy_(torch.from_numpy(vectors))
        
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
        print(f"Loading GloVe embeddings from {glove_path}...")
        
        pretrained = np.random.uniform(-0.25, 0.25, (self.vocab_size, self.embedding_dim))
        
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
        
        self.load_pretrained(pretrained, freeze=freeze)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embeddings = self.embedding(token_ids)
        
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
