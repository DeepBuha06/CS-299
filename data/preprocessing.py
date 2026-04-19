

import re
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class Preprocessor:
    
    def __init__(
        self,
        vocab: Dict[str, int],
        max_length: int = 256,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>"
    ):
        self.vocab = vocab
        self.max_length = max_length
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_idx = vocab.get(pad_token, 0)
        self.unk_idx = vocab.get(unk_token, 1)
        
        self.idx_to_token = {idx: token for token, idx in vocab.items()}
    
    @classmethod
    def from_vocab_file(cls, vocab_path: str, max_length: int = 256) -> "Preprocessor":
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return cls(vocab, max_length)
    
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        
        text = re.sub(r'<[^>]+>', ' ', text)
        
        tokens = re.findall(r'\b[a-z]+\b', text)
        
        return tokens
    
    def numericalize(self, tokens: List[str]) -> List[int]:
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
        original_length = len(indices)
        
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
            actual_length = self.max_length
        else:
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
        tokens = self.tokenize(text)
        
        indices = self.numericalize(tokens)
        
        if return_length:
            padded, length = self.pad_sequence(indices, return_length=True)
            return torch.tensor(padded, dtype=torch.long), length
        else:
            padded = self.pad_sequence(indices, return_length=False)
            return torch.tensor(padded, dtype=torch.long)
    
    def decode(self, indices: List[int], skip_special: bool = True) -> str:
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
        return len(self.vocab)


def create_attention_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    batch_size = lengths.size(0)
    range_tensor = torch.arange(max_length).unsqueeze(0).expand(batch_size, -1)
    mask = range_tensor < lengths.unsqueeze(1)
    return mask
