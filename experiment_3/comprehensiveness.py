import torch
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path

from config import ExperimentConfig


class ComprehensivenessTester:
    """Compute comprehensiveness score for attention weights."""
    
    def __init__(self, model, preprocessor):
        """
        Initialize comprehensiveness tester.
        
        Args:
            model: Trained attention model
            preprocessor: Text preprocessor
        """
        self.model = model
        self.preprocessor = preprocessor
        self.device = ExperimentConfig.DEVICE
    
    def get_baseline_prediction(self, text: str) -> Dict:
        """
        Get baseline prediction, attention scores, and hidden states.
        
        Args:
            text: Input review text
            
        Returns:
            Dictionary with:
                - tokens: List of tokens
                - token_ids: Tensor of token indices
                - prediction: Model output probability
                - attention_weights: Attention scores per token
                - hidden_states: BiLSTM hidden states (batch_size, seq_length, hidden_dim)
                - length: Actual sequence length (excluding padding)
        """
        # Tokenize
        tokens = self.preprocessor.tokenize(text)
        
        # Process
        processed = self.preprocessor.process(text, return_length=True)
        if isinstance(processed, tuple):
            token_ids, length = processed
        else:
            token_ids = processed
            length = len(tokens)
        
        # Add batch dimension
        token_ids = token_ids.unsqueeze(0).to(self.device)
        lengths = torch.tensor([length], device=self.device)
        
        # Get hidden states from encoder
        with torch.no_grad():
            embeddings = self.model.embedding(token_ids)
            hidden_states, _ = self.model.encoder(embeddings, lengths)
            
            # Create attention mask
            mask = None
            if lengths is not None:
                seq_length = token_ids.shape[1]
                range_tensor = torch.arange(seq_length, device=token_ids.device)
                range_tensor = range_tensor.unsqueeze(0).expand(1, -1)
                mask = range_tensor < lengths.unsqueeze(1)
            
            # Get attention and prediction
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
        """
        Get prediction for modified text using stored hidden states.
        
        Remove hidden states for top-k tokens and recompute attention.
        
        Args:
            removed_indices: Indices of tokens to remove
            hidden_states: Stored hidden states from original text (1, seq_len, hidden_dim)
            original_length: Original sequence length (actual, not padded)
            
        Returns:
            Model output probability after removing tokens
        """
        # Extract only the actual sequence (remove padding)
        hidden_states = hidden_states.to(self.device)
        actual_hidden = hidden_states[0, :original_length, :]  # (original_length, hidden_dim)
        
        # Create mask for remaining tokens (True = keep, False = remove)
        remaining_mask = torch.ones(original_length, dtype=torch.bool)
        for idx in removed_indices:
            remaining_mask[idx] = False
        
        # Extract hidden states for remaining tokens
        remaining_hidden = actual_hidden[remaining_mask, :].unsqueeze(0)  # (1, remaining_len, hidden_dim)
        
        # Create attention mask for remaining tokens
        remaining_length = remaining_hidden.shape[1]
        mask = torch.ones(1, remaining_length, dtype=torch.bool, device=self.device)
        
        # Recompute attention on remaining hidden states
        with torch.no_grad():
            context, _ = self.model.attention(remaining_hidden, mask)
            prediction = self.model.classifier(context)
        
        return prediction.cpu().item()
    
    def compute_comprehensiveness(
        self,
        text: str,
        top_k: int = 5
    ) -> Dict:
        """
        Compute comprehensiveness metric for a single review using stored hidden states.
        
        Steps:
        1. Get baseline prediction, attention scores, and hidden states
        2. Identify top-k high-attention tokens (get their indices)
        3. Extract hidden states for remaining tokens
        4. Recompute attention on remaining hidden states
        5. Calculate comprehensiveness = m(xi)_j - m(xi/ri)_j
        
        Args:
            text: Input review
            top_k: Number of top attended tokens to remove
            
        Returns:
            Dictionary with results:
                - original_prediction: m(xi)_j
                - modified_prediction: m(xi/ri)_j
                - comprehensiveness: m(xi)_j - m(xi/ri)_j
                - original_tokens: List of original tokens
                - attention_weights: Attention scores
                - removed_tokens: Top-k removed tokens
                - removed_indices: Indices of removed tokens
        """
        # Step 1: Get baseline with hidden states
        baseline = self.get_baseline_prediction(text)
        original_pred = baseline["prediction"]
        attention_weights = baseline["attention_weights"]
        tokens = baseline["tokens"]
        hidden_states = baseline["hidden_states"]
        original_length = baseline["length"]
        
        # Step 2: Get indices and tokens of top-k
        top_k_indices = np.argsort(attention_weights)[-top_k:]
        removed_tokens = [tokens[i] for i in top_k_indices]
        removed_scores = attention_weights[top_k_indices]
        
        # Step 3 & 4: Remove top-k and get new prediction using stored hidden states
        modified_pred = self.get_modified_prediction(
            removed_indices=list(top_k_indices),
            hidden_states=hidden_states,
            original_length=original_length
        )
        
        # Step 5: Calculate comprehensiveness
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
        """
        Compute comprehensiveness for multiple k values.
        
        Args:
            text: Input review
            k_values: List of k values to test (default: [1, 5, 10])
            
        Returns:
            Dictionary with results for each k value
        """
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
