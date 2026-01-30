"""
Encoder module (BiLSTM).
Step 4: Process embeddings through recurrent neural network to get hidden states.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    
    Step 4: Encoder
    - Takes embedding sequence [x₁, x₂, ..., xₙ]
    - Produces hidden states [h₁, h₂, ..., hₙ]
    - Each hᵢ captures both left and right context (bidirectional)
    
    Mathematical representation:
        Forward:  h→ᵢ = LSTM→(xᵢ, h→ᵢ₋₁)
        Backward: h←ᵢ = LSTM←(xᵢ, h←ᵢ₊₁)
        Combined: hᵢ = [h→ᵢ ; h←ᵢ]
        
    Where:
        - h→ᵢ is forward hidden state
        - h←ᵢ is backward hidden state
        - [; ] denotes concatenation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize BiLSTM encoder.
        
        Args:
            input_dim: Input dimension (embedding_dim)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout between LSTM layers (only if num_layers > 1)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Calculate output dimension
        self.output_dim = hidden_dim * self.num_directions
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Additional dropout for output
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through BiLSTM.
        
        Args:
            embeddings: Input embeddings
                       Shape: (batch_size, seq_length, input_dim)
            lengths: Actual sequence lengths for packing
                    Shape: (batch_size,)
        
        Returns:
            hidden_states: All hidden states
                          Shape: (batch_size, seq_length, hidden_dim * num_directions)
            (final_hidden, final_cell): Final hidden and cell states
                          Each has shape: (num_layers * num_directions, batch_size, hidden_dim)
        """
        batch_size = embeddings.size(0)
        
        if lengths is not None:
            # Pack sequences for efficient processing
            # This tells LSTM to ignore padding tokens
            
            # Sort by length (required for pack_padded_sequence)
            lengths_cpu = lengths.cpu()
            sorted_lengths, sorted_idx = lengths_cpu.sort(descending=True)
            sorted_embeddings = embeddings[sorted_idx]
            
            # Pack
            packed = pack_padded_sequence(
                sorted_embeddings,
                sorted_lengths.tolist(),
                batch_first=True
            )
            
            # Forward through LSTM
            packed_output, (hidden, cell) = self.lstm(packed)
            
            # Unpack
            hidden_states, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=embeddings.size(1)
            )
            
            # Unsort to restore original order
            _, unsorted_idx = sorted_idx.sort()
            hidden_states = hidden_states[unsorted_idx]
            
            # Also unsort the final states
            hidden = hidden[:, unsorted_idx, :]
            cell = cell[:, unsorted_idx, :]
        else:
            # No length information - process all positions
            hidden_states, (hidden, cell) = self.lstm(embeddings)
        
        # Apply dropout if specified
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        
        return hidden_states, (hidden, cell)
    
    def get_final_hidden(
        self,
        hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Get final hidden representation by concatenating forward and backward.
        
        Args:
            hidden: Final hidden state from LSTM
                   Shape: (num_layers * num_directions, batch_size, hidden_dim)
        
        Returns:
            Combined hidden state
            Shape: (batch_size, hidden_dim * num_directions)
        """
        if self.bidirectional:
            # hidden has shape (num_layers * 2, batch_size, hidden_dim)
            # Get last layer: forward is at -2, backward is at -1
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
