"""
Transformer-based Sentiment Classifier using DistilBERT.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class TransformerClassifier(nn.Module):
    """
    Sentiment classifier using DistilBERT as the backbone.
    
    Architecture:
    1. DistilBERT encoder (pre-trained)
    2. Dropout layer
    3. Linear classification head
    
    Features:
    - Can extract attention weights for visualization
    - Supports both CPU and GPU inference
    - Can be fine-tuned or used with frozen backbone
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Initialize the transformer classifier.
        
        Args:
            model_name: HuggingFace model name to load
            num_labels: Number of output classes (2 for binary)
            dropout: Dropout probability before classifier
            freeze_backbone: Whether to freeze DistilBERT weights
        """
        super().__init__()
        
        self.num_labels = num_labels
        
        # Load pre-trained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(
            model_name,
            output_attentions=True  # Enable attention output
        )
        
        # Optionally freeze DistilBERT weights
        if freeze_backbone:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        # Get hidden size from config
        self.hidden_size = self.distilbert.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Classification logits (batch_size, num_labels)
            attention: Attention weights if return_attention=True
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attention
        )
        
        # Use [CLS] token representation (first token)
        # outputs.last_hidden_state: (batch_size, seq_length, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Apply dropout and classifier
        pooled_output = self.dropout(cls_output)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)
        
        if return_attention:
            # Get attention from all layers, use last layer
            # attentions is tuple of (batch, num_heads, seq_len, seq_len)
            attentions = outputs.attentions
            # Average across heads from last layer
            last_layer_attention = attentions[-1]  # (batch, num_heads, seq, seq)
            avg_attention = last_layer_attention.mean(dim=1)  # (batch, seq, seq)
            # Get attention from [CLS] token to other tokens
            cls_attention = avg_attention[:, 0, :]  # (batch, seq)
            return logits, cls_attention
        
        return logits, None
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            
        Returns:
            Probabilities (batch_size, num_labels)
        """
        logits, _ = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)
    
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: int = -1,
        head: int = None
    ) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            layer: Which transformer layer (-1 for last)
            head: Which attention head (None for average across heads)
            
        Returns:
            Attention weights (batch_size, seq_length, seq_length)
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        # Get attention from specified layer
        attentions = outputs.attentions[layer]  # (batch, num_heads, seq, seq)
        
        if head is not None:
            return attentions[:, head, :, :]
        else:
            return attentions.mean(dim=1)  # Average across heads
    
    @classmethod
    def from_pretrained(cls, path: str, model_name: str = "distilbert-base-uncased"):
        """
        Load a fine-tuned model from checkpoint.
        
        Args:
            path: Path to saved model checkpoint
            model_name: Original model name for architecture
            
        Returns:
            Loaded TransformerClassifier
        """
        model = cls(model_name=model_name)
        checkpoint = torch.load(path, map_location="cpu")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        return model


def count_parameters(model: nn.Module) -> tuple:
    """
    Count model parameters.
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
