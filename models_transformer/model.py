
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class TransformerClassifier(nn.Module):
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        self.distilbert = DistilBertModel.from_pretrained(
            model_name,
            output_attentions=True  # Enable attention output
        )
        
        if freeze_backbone:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        self.hidden_size = self.distilbert.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False
    ):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attention
        )
        
        # outputs.last_hidden_state: (batch_size, seq_length, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        pooled_output = self.dropout(cls_output)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)
        
        if return_attention:
            # attentions is tuple of (batch, num_heads, seq_len, seq_len)
            attentions = outputs.attentions
            last_layer_attention = attentions[-1]  # (batch, num_heads, seq, seq)
            avg_attention = last_layer_attention.mean(dim=1)  # (batch, seq, seq)
            cls_attention = avg_attention[:, 0, :]  # (batch, seq)
            return logits, cls_attention
        
        return logits, None
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        logits, _ = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)
    
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: int = -1,
        head: int = None
    ) -> torch.Tensor:
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        attentions = outputs.attentions[layer]  # (batch, num_heads, seq, seq)
        
        if head is not None:
            return attentions[:, head, :, :]
        else:
            return attentions.mean(dim=1)  # Average across heads
    
    @classmethod
    def from_pretrained(cls, path: str, model_name: str = "distilbert-base-uncased"):
        model = cls(model_name=model_name)
        checkpoint = torch.load(path, map_location="cpu")
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        return model


def count_parameters(model: nn.Module) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
