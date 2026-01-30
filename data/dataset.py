"""
Dataset module for IMDB data.
Step 1: Loading raw reviews and creating PyTorch datasets.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import Preprocessor


class IMDBDataset(Dataset):
    """
    PyTorch Dataset for IMDB movie reviews.
    
    Step 1: Raw Input
    - Loads reviews from pos/neg folders
    - Associates each review with a label (0=negative, 1=positive)
    """
    
    def __init__(
        self,
        data_dir: str,
        preprocessor: Preprocessor,
        split: str = "train"
    ):
        """
        Initialize IMDB dataset.
        
        Args:
            data_dir: Root directory containing train/test folders
            preprocessor: Preprocessor instance for text processing
            split: "train" or "test"
        """
        self.data_dir = Path(data_dir) / split
        self.preprocessor = preprocessor
        self.split = split
        
        # Load all reviews
        self.reviews, self.labels = self._load_reviews()
        
        print(f"Loaded {len(self.reviews)} reviews from {split} set")
        print(f"  Positive: {sum(self.labels)}, Negative: {len(self.labels) - sum(self.labels)}")
    
    def _load_reviews(self) -> Tuple[List[str], List[int]]:
        """
        Load reviews from neg and pos folders.
        
        Returns:
            Tuple of (reviews list, labels list)
        """
        reviews = []
        labels = []
        
        # Load negative reviews (label = 0)
        neg_dir = self.data_dir / "neg"
        if neg_dir.exists():
            for file_path in neg_dir.glob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    labels.append(0)
        
        # Load positive reviews (label = 1)
        pos_dir = self.data_dir / "pos"
        if pos_dir.exists():
            for file_path in pos_dir.glob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    labels.append(1)
        
        return reviews, labels
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.reviews)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (token_ids, label, sequence_length)
        """
        review = self.reviews[idx]
        label = self.labels[idx]
        
        # Preprocess: tokenize -> numericalize -> pad
        token_ids, length = self.preprocessor.process(review, return_length=True)
        
        return token_ids, torch.tensor(label, dtype=torch.float), length


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of (token_ids, label, length) tuples
        
    Returns:
        Tuple of batched tensors:
        - token_ids: (batch_size, max_length)
        - labels: (batch_size,)
        - lengths: (batch_size,)
    """
    token_ids, labels, lengths = zip(*batch)
    
    token_ids = torch.stack(token_ids)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return token_ids, labels, lengths


def get_dataloaders(
    data_dir: str,
    preprocessor: Preprocessor,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        data_dir: Root directory containing train/test folders
        preprocessor: Preprocessor instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = IMDBDataset(data_dir, preprocessor, split="train")
    test_dataset = IMDBDataset(data_dir, preprocessor, split="test")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


class IMDBDatasetFromList(Dataset):
    """
    Alternative Dataset that loads from pre-loaded lists.
    Useful for quick experimentation or when data is already in memory.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        preprocessor: Preprocessor
    ):
        """
        Initialize from lists.
        
        Args:
            texts: List of review texts
            labels: List of labels (0 or 1)
            preprocessor: Preprocessor instance
        """
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        token_ids, length = self.preprocessor.process(text, return_length=True)
        
        return token_ids, torch.tensor(label, dtype=torch.float), length
