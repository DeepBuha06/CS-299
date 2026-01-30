"""
Training script for Attention-based Text Classifier.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from config import Config
from data.preprocessing import Preprocessor
from data.dataset import get_dataloaders
from models.model import AttentionClassifier
from utils.metrics import calculate_metrics, print_metrics


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    progress = tqdm(dataloader, desc="Training")
    for token_ids, labels, lengths in progress:
        # Move to device
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions, _ = model(token_ids, lengths)
        
        # Compute loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(labels.detach().cpu())
        
        progress.set_postfix(loss=loss.item())
    
    # Calculate epoch metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_predictions, all_targets)
    metrics["loss"] = total_loss / len(dataloader)
    
    return metrics


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for token_ids, labels, lengths in tqdm(dataloader, desc="Evaluating"):
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            predictions, _ = model(token_ids, lengths)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(labels.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_predictions, all_targets)
    metrics["loss"] = total_loss / len(dataloader)
    
    return metrics


def main():
    # Print configuration
    Config.print_config()
    
    # Set seed for reproducibility
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    preprocessor = Preprocessor.from_vocab_file(
        Config.VOCAB_FILE,
        max_length=Config.MAX_SEQ_LENGTH
    )
    print(f"Vocabulary size: {preprocessor.vocab_size}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, test_loader = get_dataloaders(
        Config.DATA_DIR,
        preprocessor,
        batch_size=Config.BATCH_SIZE
    )
    
    # Create model
    print("\nCreating model...")
    model = AttentionClassifier(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        attention_dim=Config.ATTENTION_DIM,
        num_classes=Config.NUM_CLASSES,
        num_layers=Config.NUM_LAYERS,
        bidirectional=Config.BIDIRECTIONAL,
        attention_type=Config.ATTENTION_TYPE,
        encoder_dropout=Config.ENCODER_DROPOUT,
        classifier_dropout=Config.CLASSIFIER_DROPOUT,
        padding_idx=Config.PAD_IDX
    )
    model = model.to(Config.DEVICE)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    best_accuracy = 0
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            Config.DEVICE, Config.CLIP_GRAD
        )
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print_metrics(train_metrics, prefix="Train ")
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, Config.DEVICE)
        print(f"\nTest Loss: {test_metrics['loss']:.4f}")
        print_metrics(test_metrics, prefix="Test ")
        
        # Save best model
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            torch.save(model.state_dict(), Config.MODEL_DIR / "best_model.pt")
            print(f"Saved best model (accuracy: {best_accuracy:.4f})")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best accuracy: {best_accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
