import torch
import torch.nn as nn
from typing import Tuple, Optional


class Classifier(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        if hidden_dim is not None:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.fc = nn.Linear(input_dim, num_classes)
        
        self.sigmoid = nn.Sigmoid() if num_classes == 1 else None
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        dropped = self.dropout(context)
        logits = self.fc(dropped)
        
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
            return self.sigmoid(logits)
        return torch.softmax(logits, dim=-1)
    
    def predict(self, context: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probabilities = self.forward(context)
        if self.num_classes == 1:
            return (probabilities >= threshold).long()
        return probabilities.argmax(dim=-1)


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    if predictions.dim() == 1:
        predicted_classes = (predictions >= threshold).long()
        targets = targets.long()
    else:
        predicted_classes = predictions.argmax(dim=-1)
    correct = (predicted_classes == targets).sum().item()
    return correct / targets.size(0)
