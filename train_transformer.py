"""
Training script for Transformer-based Sentiment Classifier.
Uses DistilBERT with fine-tuning on IMDB dataset.
"""

import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm

from config_transformer import TransformerConfig
from models_transformer.model import TransformerClassifier, count_parameters
from models_transformer.dataset import get_dataloaders, get_tokenizer


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: The transformer model
        dataloader: Training dataloader
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    progress = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        logits, _ = model(input_ids, attention_mask)
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Accumulate metrics
        total_loss += loss.item() * gradient_accumulation_steps
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        progress.set_postfix(loss=loss.item() * gradient_accumulation_steps, acc=correct/total)
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total
    }


def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device
) -> dict:
    """
    Evaluate on validation/test set.
    
    Args:
        model: The transformer model
        dataloader: Evaluation dataloader
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Calculate additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    # Print configuration
    TransformerConfig.print_config()
    
    # Set seed for reproducibility
    torch.manual_seed(TransformerConfig.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TransformerConfig.SEED)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer(TransformerConfig.MODEL_NAME)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, test_loader = get_dataloaders(
        TransformerConfig.DATA_DIR,
        tokenizer,
        max_length=TransformerConfig.MAX_SEQ_LENGTH,
        batch_size=TransformerConfig.BATCH_SIZE
    )
    
    # Create model
    print("\nCreating model...")
    model = TransformerClassifier(
        model_name=TransformerConfig.MODEL_NAME,
        num_labels=TransformerConfig.NUM_LABELS,
        dropout=0.1
    )
    model = model.to(TransformerConfig.DEVICE)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Calculate training steps
    num_training_steps = len(train_loader) * TransformerConfig.NUM_EPOCHS
    num_warmup_steps = int(num_training_steps * TransformerConfig.WARMUP_RATIO)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=TransformerConfig.LEARNING_RATE,
        weight_decay=TransformerConfig.WEIGHT_DECAY
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    best_accuracy = 0
    best_metrics = None
    
    for epoch in range(TransformerConfig.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{TransformerConfig.NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            TransformerConfig.DEVICE,
            TransformerConfig.GRADIENT_ACCUMULATION_STEPS,
            TransformerConfig.MAX_GRAD_NORM
        )
        print(f"Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, TransformerConfig.DEVICE)
        print(f"Test Loss: {test_metrics['loss']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        # Save best model
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            best_metrics = test_metrics
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "accuracy": best_accuracy
            }
            torch.save(
                checkpoint,
                TransformerConfig.MODEL_DIR / TransformerConfig.MODEL_CHECKPOINT
            )
            print(f"Saved best model (accuracy: {best_accuracy:.4f})")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best accuracy: {best_accuracy:.4f}")
    print("=" * 60)
    
    # Save final metrics
    if best_metrics:
        metrics_to_save = {
            "test_accuracy": float(best_metrics["accuracy"]),
            "test_loss": float(best_metrics["loss"]),
            "test_precision": float(best_metrics["precision"]),
            "test_recall": float(best_metrics["recall"]),
            "test_f1": float(best_metrics["f1"]),
            "model": TransformerConfig.MODEL_NAME
        }
        with open(TransformerConfig.MODEL_DIR / TransformerConfig.METRICS_FILE, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"Saved metrics to {TransformerConfig.MODEL_DIR / TransformerConfig.METRICS_FILE}")


if __name__ == "__main__":
    main()
