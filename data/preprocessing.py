"""
Preprocessing module for text data.
Step 2: Tokenization, numericalization, and padding.
"""

import re
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class Preprocessor:
    """
    Text preprocessor that handles:
    - Tokenization (splitting text into tokens)
    - Numericalization (converting tokens to indices)
    - Padding/truncation (making sequences same length)
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        max_length: int = 256,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>"
    ):
        """
        Initialize preprocessor.
        
        Args:
            vocab: Dictionary mapping tokens to indices
            max_length: Maximum sequence length
            pad_token: Padding token string
            unk_token: Unknown token string
        """
        self.vocab = vocab
        self.max_length = max_length
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_idx = vocab.get(pad_token, 0)
        self.unk_idx = vocab.get(unk_token, 1)
        
        # Create reverse vocabulary for decoding
        self.idx_to_token = {idx: token for token, idx in vocab.items()}
    
    @classmethod
    def from_vocab_file(cls, vocab_path: str, max_length: int = 256) -> "Preprocessor":
        """
        Create preprocessor from a vocabulary JSON file.
        
        Args:
            vocab_path: Path to vocab.json file
            max_length: Maximum sequence length
            
        Returns:
            Preprocessor instance
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return cls(vocab, max_length)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into list of tokens.
        
        Step 2a: Tokenization
        - Convert to lowercase
        - Remove HTML tags
        - Extract alphabetic words
        
        Args:
            text: Raw input text
            
        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()
        
        # Remove HTML tags (like <br />)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Extract words (alphabetic sequences only)
        tokens = re.findall(r'\b[a-z]+\b', text)
        
        return tokens
    
    def numericalize(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to their vocabulary indices.
        
        Step 2b: Numericalization
        - Look up each token in vocabulary
        - Use UNK index for out-of-vocabulary tokens
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token indices
        """
        indices = []
        for token in tokens:
            idx = self.vocab.get(token, self.unk_idx)
            indices.append(idx)
        return indices
    
    def pad_sequence(
        self,
        indices: List[int],
        return_length: bool = False
    ) -> Tuple[List[int], int] | List[int]:
        """
        Pad or truncate sequence to fixed length.
        
        Step 2c: Padding/Truncation
        - Truncate if longer than max_length
        - Pad with PAD index if shorter
        
        Args:
            indices: List of token indices
            return_length: Whether to return original length
            
        Returns:
            Padded sequence (and optionally original length)
        """
        original_length = len(indices)
        
        if len(indices) > self.max_length:
            # Truncate
            indices = indices[:self.max_length]
            actual_length = self.max_length
        else:
            # Pad
            actual_length = len(indices)
            padding = [self.pad_idx] * (self.max_length - len(indices))
            indices = indices + padding
        
        if return_length:
            return indices, actual_length
        return indices
    
    def process(
        self,
        text: str,
        return_length: bool = False
    ) -> Tuple[torch.Tensor, int] | torch.Tensor:
        """
        Full preprocessing pipeline: tokenize -> numericalize -> pad.
        
        Args:
            text: Raw input text
            return_length: Whether to return original sequence length
            
        Returns:
            Tensor of token indices (and optionally length)
        """
        # Step 2a: Tokenize
        tokens = self.tokenize(text)
        
        # Step 2b: Numericalize
        indices = self.numericalize(tokens)
        
        # Step 2c: Pad/truncate
        if return_length:
            padded, length = self.pad_sequence(indices, return_length=True)
            return torch.tensor(padded, dtype=torch.long), length
        else:
            padded = self.pad_sequence(indices, return_length=False)
            return torch.tensor(padded, dtype=torch.long)
    
    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Convert indices back to text.
        
        Args:
            indices: List of token indices
            skip_special: Whether to skip PAD tokens
            
        Returns:
            Decoded text string
        """
        tokens = []
        for idx in indices:
            if skip_special and idx == self.pad_idx:
                continue
            token = self.idx_to_token.get(idx, self.unk_token)
            tokens.append(token)
        return " ".join(tokens)
    
    def batch_process(
        self,
        texts: List[str],
        return_lengths: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Process a batch of texts.
        
        Args:
            texts: List of raw text strings
            return_lengths: Whether to return sequence lengths
            
        Returns:
            Batch tensor of shape (batch_size, max_length)
            Optionally also lengths tensor of shape (batch_size,)
        """
        batch_indices = []
        lengths = []
        
        for text in texts:
            if return_lengths:
                indices, length = self.process(text, return_length=True)
                lengths.append(length)
            else:
                indices = self.process(text, return_length=False)
            batch_indices.append(indices)
        
        batch_tensor = torch.stack(batch_indices)
        
        if return_lengths:
            lengths_tensor = torch.tensor(lengths, dtype=torch.long)
            return batch_tensor, lengths_tensor
        
        return batch_tensor
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)


def create_attention_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Create attention mask to ignore padding tokens.
    
    Args:
        lengths: Tensor of actual sequence lengths (batch_size,)
        max_length: Maximum sequence length
        
    Returns:
        Boolean mask of shape (batch_size, max_length)
        True for valid tokens, False for padding
    """
    batch_size = lengths.size(0)
    # Create range tensor [0, 1, 2, ..., max_length-1]
    range_tensor = torch.arange(max_length).unsqueeze(0).expand(batch_size, -1)
    # Compare with lengths
    mask = range_tensor < lengths.unsqueeze(1)
    return mask
