import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from .embeddings import EmbeddingLayer
from .encoder import BiLSTMEncoder
from .attention import AdditiveAttention, DotProductAttention, create_attention
from .classifier import Classifier


class AttentionClassifier(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        num_classes: int = 1,
        num_layers: int = 1,
        bidirectional: bool = True,
        attention_type: str = "additive",
        embedding_dropout: float = 0.0,
        encoder_dropout: float = 0.3,
        classifier_dropout: float = 0.5,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            dropout=embedding_dropout
        )
        
        self.encoder = BiLSTMEncoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=encoder_dropout
        )
        
        encoder_output_dim = self.encoder.output_dim
        
        self.attention = create_attention(
            attention_type=attention_type,
            hidden_dim=encoder_output_dim,
            attention_dim=attention_dim
        )
        
        self.classifier = Classifier(
            input_dim=encoder_output_dim,
            num_classes=num_classes,
            dropout=classifier_dropout
        )
    
    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length = token_ids.shape
        
        mask = None
        if lengths is not None:
            range_tensor = torch.arange(seq_length, device=token_ids.device)
            range_tensor = range_tensor.unsqueeze(0).expand(batch_size, -1)
            mask = range_tensor < lengths.unsqueeze(1)
        
        embeddings = self.embedding(token_ids)
        
        hidden_states, _ = self.encoder(embeddings, lengths)
        
        context, attention_weights = self.attention(hidden_states, mask)
        
        predictions = self.classifier(context)
        
        if return_attention:
            return predictions, attention_weights
        return predictions, None
    
    def predict(
        self,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> torch.Tensor:
        predictions, _ = self.forward(token_ids, lengths)
        if self.num_classes == 1:
            return (predictions >= threshold).long()
        return predictions.argmax(dim=-1)
    
    def get_attention_weights(
        self,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, attention_weights = self.forward(token_ids, lengths, return_attention=True)
        return attention_weights


def create_model_from_config(config) -> AttentionClassifier:
    return AttentionClassifier(
        vocab_size=config.VOCAB_SIZE + 2,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        attention_dim=config.ATTENTION_DIM,
        num_classes=config.NUM_CLASSES,
        num_layers=config.NUM_LAYERS,
        bidirectional=config.BIDIRECTIONAL,
        attention_type=config.ATTENTION_TYPE,
        encoder_dropout=config.ENCODER_DROPOUT,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
        padding_idx=config.PAD_IDX
    )
