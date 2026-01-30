"""
Metrics module for evaluation.
"""

import torch
from typing import Dict, List, Tuple
from collections import defaultdict


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Predicted probabilities
        targets: True labels
        threshold: Decision threshold
    
    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    # Convert probabilities to class predictions
    if predictions.dim() == 1:
        pred_classes = (predictions >= threshold).long()
    else:
        pred_classes = predictions.argmax(dim=-1)
    
    targets = targets.long()
    
    # True positives, false positives, etc.
    tp = ((pred_classes == 1) & (targets == 1)).sum().item()
    fp = ((pred_classes == 1) & (targets == 0)).sum().item()
    tn = ((pred_classes == 0) & (targets == 0)).sum().item()
    fn = ((pred_classes == 0) & (targets == 1)).sum().item()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Print metrics in a formatted way."""
    print(f"{prefix}Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall:    {metrics['recall']:.4f}")
    print(f"{prefix}F1 Score:  {metrics['f1']:.4f}")


class MetricTracker:
    """Track metrics over training."""
    
    def __init__(self):
        self.history = defaultdict(list)
    
    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            self.history[key].append(value)
    
    def get_best(self, metric: str = "accuracy") -> Tuple[int, float]:
        values = self.history.get(metric, [])
        if not values:
            return -1, 0.0
        best_idx = max(range(len(values)), key=lambda i: values[i])
        return best_idx, values[best_idx]
