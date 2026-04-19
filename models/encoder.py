import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional


class BiLSTMEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.output_dim = hidden_dim * self.num_directions
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = embeddings.size(0)
        
        if lengths is not None:
            
            lengths_cpu = lengths.cpu()
            sorted_lengths, sorted_idx = lengths_cpu.sort(descending=True)
            sorted_embeddings = embeddings[sorted_idx]
            
            packed = pack_padded_sequence(
                sorted_embeddings,
                sorted_lengths.tolist(),
                batch_first=True
            )
            
            packed_output, (hidden, cell) = self.lstm(packed)
            
            hidden_states, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=embeddings.size(1)
            )
            
            _, unsorted_idx = sorted_idx.sort()
            hidden_states = hidden_states[unsorted_idx]
            
            hidden = hidden[:, unsorted_idx, :]
            cell = cell[:, unsorted_idx, :]
        else:
            hidden_states, (hidden, cell) = self.lstm(embeddings)
        
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        
        return hidden_states, (hidden, cell)
    
    def get_final_hidden(
        self,
        hidden: torch.Tensor
    ) -> torch.Tensor:
        if self.bidirectional:
            # hidden has shape (num_layers * 2, batch_size, hidden_dim)
            forward = hidden[-2, :, :]  # (batch_size, hidden_dim)
            backward = hidden[-1, :, :]  # (batch_size, hidden_dim)
            combined = torch.cat([forward, backward], dim=1)
        else:
            combined = hidden[-1, :, :]
        
        return combined
    
    def __repr__(self):
        return (
            f"BiLSTMEncoder("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}, "
            f"num_layers={self.num_layers}, "
            f"bidirectional={self.bidirectional})"
        )
