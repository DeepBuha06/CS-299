import torch
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path

from config import ExperimentConfig


class ComprehensivenessTester:
    
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.device = ExperimentConfig.DEVICE
    
    def get_baseline_prediction(self, text: str) -> Dict:
        tokens = self.preprocessor.tokenize(text)
        
        processed = self.preprocessor.process(text, return_length=True)
        if isinstance(processed, tuple):
            token_ids, length = processed
        else:
            token_ids = processed
            length = len(tokens)
        
        token_ids = token_ids.unsqueeze(0).to(self.device)
        lengths = torch.tensor([length], device=self.device)
        
        with torch.no_grad():
            embeddings = self.model.embedding(token_ids)
            hidden_states, _ = self.model.encoder(embeddings, lengths)
            
            mask = None
            if lengths is not None:
                seq_length = token_ids.shape[1]
                range_tensor = torch.arange(seq_length, device=token_ids.device)
                range_tensor = range_tensor.unsqueeze(0).expand(1, -1)
                mask = range_tensor < lengths.unsqueeze(1)
            
            context, attention_weights = self.model.attention(hidden_states, mask)
            prediction = self.model.classifier(context)
        
        prediction_prob = prediction.cpu().item()
        attention_scores = attention_weights[0, :length].cpu().numpy()
        
        return {
            "tokens": tokens[:length],
            "token_ids": token_ids.cpu(),
            "prediction": prediction_prob,
            "attention_weights": attention_scores,
            "hidden_states": hidden_states.cpu(),  # Store for reuse
            "length": length
        }
    
    def get_modified_prediction(
        self,
        removed_indices: list,
        hidden_states: torch.Tensor,
        original_length: int
    ) -> float:
        hidden_states = hidden_states.to(self.device)
        actual_hidden = hidden_states[0, :original_length, :]  # (original_length, hidden_dim)
        
        remaining_mask = torch.ones(original_length, dtype=torch.bool)
        for idx in removed_indices:
            remaining_mask[idx] = False
        
        remaining_hidden = actual_hidden[remaining_mask, :].unsqueeze(0)  # (1, remaining_len, hidden_dim)
        
        remaining_length = remaining_hidden.shape[1]
        mask = torch.ones(1, remaining_length, dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            context, _ = self.model.attention(remaining_hidden, mask)
            prediction = self.model.classifier(context)
        
        return prediction.cpu().item()
    
    def compute_comprehensiveness(
        self,
        text: str,
        top_k: int = 5
    ) -> Dict:
        baseline = self.get_baseline_prediction(text)
        original_pred = baseline["prediction"]
        attention_weights = baseline["attention_weights"]
        tokens = baseline["tokens"]
        hidden_states = baseline["hidden_states"]
        original_length = baseline["length"]
        
        top_k_indices = np.argsort(attention_weights)[-top_k:]
        removed_tokens = [tokens[i] for i in top_k_indices]
        removed_scores = attention_weights[top_k_indices]
        
        modified_pred = self.get_modified_prediction(
            removed_indices=list(top_k_indices),
            hidden_states=hidden_states,
            original_length=original_length
        )
        
        comprehensiveness = original_pred - modified_pred
        
        return {
            "original_text": text,
            "original_prediction": original_pred,
            "modified_prediction": modified_pred,
            "comprehensiveness": comprehensiveness,
            "top_k": top_k,
            "original_tokens": tokens,
            "attention_weights": attention_weights.tolist(),
            "removed_tokens": removed_tokens,
            "removed_scores": removed_scores.tolist(),
            "removed_indices": list(map(int, top_k_indices))
        }
    
    def compute_multiple_k(
        self,
        text: str,
        k_values: List[int] = None
    ) -> Dict:
        if k_values is None:
            k_values = ExperimentConfig.TOP_K_VALUES
        
        results = {
            "original_text": text,
            "results_by_k": {}
        }
        
        for k in k_values:
            result = self.compute_comprehensiveness(text, top_k=k)
            results["results_by_k"][k] = result
        
        return results
