import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Attention(nn.Module):
    # eᵢ = f(hᵢ), αᵢ = softmax(eᵢ), c = Σᵢ αᵢ · hᵢ
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement forward()")


class AdditiveAttention(Attention):
    # eᵢ = vᵀ · tanh(Wₕ · hᵢ + b)
    
    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int = 128
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        self.W_h = nn.Linear(hidden_dim, attention_dim, bias=True)
        
        # v: attention_dim -> 1
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, attention_dim) -> scalar scores
        projected = self.W_h(hidden_states)
        projected = torch.tanh(projected)
        scores = self.v(projected).squeeze(-1)
        
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 1.0 / hidden_states.size(1)
        )
        
        weights_expanded = attention_weights.unsqueeze(-1)
        weighted = hidden_states * weights_expanded
        context = weighted.sum(dim=1)
        
        return context, attention_weights
    
    def __repr__(self):
        return f"AdditiveAttention(hidden_dim={self.hidden_dim}, attention_dim={self.attention_dim})"


class DotProductAttention(Attention):
    # eᵢ = (qᵀ · hᵢ) / √d
    
    def __init__(
        self,
        hidden_dim: int,
        scale: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.scale = scale
        
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.scale_factor = hidden_dim ** 0.5 if scale else 1.0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(hidden_states, self.query)
        scores = scores / self.scale_factor
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 1.0 / hidden_states.size(1)
        )
        
        
        weights_expanded = attention_weights.unsqueeze(-1)
        weighted = hidden_states * weights_expanded
        context = weighted.sum(dim=1)
        
        return context, attention_weights
    
    def __repr__(self):
        return f"DotProductAttention(hidden_dim={self.hidden_dim}, scale={self.scale})"


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** 0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        
        Q = self.W_q(hidden_states)
        K = self.W_k(hidden_states)
        V = self.W_v(hidden_states)
        
        # (batch, seq, hidden) -> (batch, heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        output = self.W_o(output)
        
        return output, attention_weights


def create_attention(
    attention_type: str,
    hidden_dim: int,
    attention_dim: int = 128
) -> Attention:
    if attention_type == "additive":
        return AdditiveAttention(hidden_dim, attention_dim)
    elif attention_type == "dot":
        return DotProductAttention(hidden_dim)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
