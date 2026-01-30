"""
Attention mechanism module.
Steps 5, 6, 7: Compute attention scores, normalize with softmax, create context vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Attention(nn.Module):
    """
    Base attention class.
    
    Steps 5-7: Attention Mechanism
    
    Step 5: Compute attention scores
        eᵢ = f(hᵢ)    where f is a scoring function
        
    Step 6: Softmax normalization
        αᵢ = exp(eᵢ) / Σⱼexp(eⱼ)    such that Σᵢαᵢ = 1
        
    Step 7: Context vector computation
        c = Σᵢ αᵢ · hᵢ    (weighted sum of hidden states)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention and context vector.
        
        Args:
            hidden_states: Encoder hidden states
                          Shape: (batch_size, seq_length, hidden_dim)
            mask: Boolean mask for valid positions (True = valid, False = padding)
                 Shape: (batch_size, seq_length)
        
        Returns:
            context: Weighted context vector
                    Shape: (batch_size, hidden_dim)
            attention_weights: Attention distribution
                              Shape: (batch_size, seq_length)
        """
        raise NotImplementedError("Subclasses must implement forward()")


class AdditiveAttention(Attention):
    """
    Additive (Bahdanau) Attention.
    
    Scoring function:
        eᵢ = vᵀ · tanh(Wₕ · hᵢ + b)
        
    Where:
        - Wₕ ∈ ℝ^{attention_dim × hidden_dim}
        - v ∈ ℝ^{attention_dim}
        - b ∈ ℝ^{attention_dim}
    """
    
    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int = 128
    ):
        """
        Initialize additive attention.
        
        Args:
            hidden_dim: Dimension of hidden states
            attention_dim: Dimension of attention intermediate representation
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Linear projection: hidden_dim -> attention_dim
        self.W_h = nn.Linear(hidden_dim, attention_dim, bias=True)
        
        # Attention vector: attention_dim -> 1
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute additive attention.
        
        Args:
            hidden_states: Shape (batch_size, seq_length, hidden_dim)
            mask: Shape (batch_size, seq_length), True for valid positions
        
        Returns:
            context: Shape (batch_size, hidden_dim)
            attention_weights: Shape (batch_size, seq_length)
        """
        # Step 5: Compute attention scores
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, attention_dim)
        projected = self.W_h(hidden_states)
        
        # Apply tanh activation
        projected = torch.tanh(projected)
        
        # Compute scalar scores
        # (batch, seq_len, attention_dim) -> (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.v(projected).squeeze(-1)
        
        # Step 6: Apply mask and softmax
        if mask is not None:
            # Set padding positions to large negative value (will become ~0 after softmax)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle case where all positions are masked
        # Replace NaN with uniform distribution
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 1.0 / hidden_states.size(1)
        )
        
        # Step 7: Compute context vector (weighted sum)
        # (batch, seq_len) -> (batch, seq_len, 1)
        weights_expanded = attention_weights.unsqueeze(-1)
        
        # (batch, seq_len, hidden_dim) * (batch, seq_len, 1) -> (batch, seq_len, hidden_dim)
        weighted = hidden_states * weights_expanded
        
        # Sum over sequence dimension
        # (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
        context = weighted.sum(dim=1)
        
        return context, attention_weights
    
    def __repr__(self):
        return f"AdditiveAttention(hidden_dim={self.hidden_dim}, attention_dim={self.attention_dim})"


class DotProductAttention(Attention):
    """
    Dot-Product Attention.
    
    Scoring function:
        eᵢ = qᵀ · hᵢ
        
    Where q is a learned query vector.
    
    Alternatively, can use scaled dot-product:
        eᵢ = (qᵀ · hᵢ) / √d
    """
    
    def __init__(
        self,
        hidden_dim: int,
        scale: bool = True
    ):
        """
        Initialize dot-product attention.
        
        Args:
            hidden_dim: Dimension of hidden states
            scale: Whether to scale by sqrt(hidden_dim)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.scale = scale
        
        # Learned query vector
        self.query = nn.Parameter(torch.randn(hidden_dim))
        
        # Scale factor
        self.scale_factor = hidden_dim ** 0.5 if scale else 1.0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dot-product attention.
        
        Args:
            hidden_states: Shape (batch_size, seq_length, hidden_dim)
            mask: Shape (batch_size, seq_length), True for valid positions
        
        Returns:
            context: Shape (batch_size, hidden_dim)
            attention_weights: Shape (batch_size, seq_length)
        """
        # Step 5: Compute attention scores via dot product
        # query: (hidden_dim,)
        # hidden_states: (batch, seq_len, hidden_dim)
        # result: (batch, seq_len)
        scores = torch.matmul(hidden_states, self.query)
        
        # Scale
        scores = scores / self.scale_factor
        
        # Step 6: Apply mask and softmax
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle NaN
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 1.0 / hidden_states.size(1)
        )
        
        # Step 7: Compute context vector
        weights_expanded = attention_weights.unsqueeze(-1)
        weighted = hidden_states * weights_expanded
        context = weighted.sum(dim=1)
        
        return context, attention_weights
    
    def __repr__(self):
        return f"DotProductAttention(hidden_dim={self.hidden_dim}, scale={self.scale})"


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (Transformer-style).
    
    More advanced attention mechanism where:
    - Multiple attention heads capture different aspects
    - Uses Q, K, V projections
    
    Note: This is more complex than what's typically used in the 
    "Attention is Not Explanation" paper, but included for completeness.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_dim: Input/output dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Q, K, V projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** 0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head self-attention.
        
        Args:
            hidden_states: Shape (batch_size, seq_length, hidden_dim)
            mask: Shape (batch_size, seq_length)
        
        Returns:
            output: Shape (batch_size, seq_length, hidden_dim)
            attention_weights: Shape (batch_size, num_heads, seq_length, seq_length)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        Q = self.W_q(hidden_states)
        K = self.W_k(hidden_states)
        V = self.W_v(hidden_states)
        
        # Reshape for multi-head: (batch, seq, hidden) -> (batch, heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (batch, heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        output = self.W_o(output)
        
        return output, attention_weights


def create_attention(
    attention_type: str,
    hidden_dim: int,
    attention_dim: int = 128
) -> Attention:
    """
    Factory function to create attention layer.
    
    Args:
        attention_type: "additive" or "dot"
        hidden_dim: Dimension of hidden states
        attention_dim: Dimension for additive attention
    
    Returns:
        Attention module
    """
    if attention_type == "additive":
        return AdditiveAttention(hidden_dim, attention_dim)
    elif attention_type == "dot":
        return DotProductAttention(hidden_dim)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
