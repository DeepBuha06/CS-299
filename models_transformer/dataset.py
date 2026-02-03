"""
Dataset module for Transformer-based sentiment analysis.
Uses HuggingFace tokenizer for preprocessing.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer


class IMDBTransformerDataset(Dataset):
    """
    PyTorch Dataset for IMDB movie reviews using HuggingFace tokenizer.
    
    Unlike the BiLSTM version which uses a custom vocabulary,
    this uses the DistilBERT tokenizer for subword tokenization.
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: DistilBertTokenizer,
        max_length: int = 512,
        split: str = "train"
    ):
        """
        Initialize IMDB dataset for transformer.
        
        Args:
            data_dir: Root directory containing train/test folders
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            split: "train" or "test"
        """
        self.data_dir = Path(data_dir) / split
        self.tokenizer = tokenizer
        self.max_length = max_length
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, and label
        """
        review = self.reviews[idx]
        label = self.labels[idx]
        
        # Tokenize using HuggingFace tokenizer
        encoding = self.tokenizer(
            review,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of dictionaries from __getitem__
        
    Returns:
        Batched tensors dictionary
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def get_dataloaders(
    data_dir: str,
    tokenizer: DistilBertTokenizer,
    max_length: int = 512,
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        data_dir: Root directory containing train/test folders
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = IMDBTransformerDataset(
        data_dir, tokenizer, max_length, split="train"
    )
    test_dataset = IMDBTransformerDataset(
        data_dir, tokenizer, max_length, split="test"
    )
    
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


def get_tokenizer(model_name: str = "distilbert-base-uncased") -> DistilBertTokenizer:
    """
    Load the DistilBERT tokenizer.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        DistilBertTokenizer instance
    """
    return DistilBertTokenizer.from_pretrained(model_name)
