import torch
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path

from config import ExperimentConfig


class SufficiencyTester:
    """Compute sufficiency score for attention weights."""
    
    def __init__(self, model, preprocessor):
        """
        Initialize sufficiency tester.
        
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
    
    def get_kept_prediction(
        self,
        kept_indices: list,
        hidden_states: torch.Tensor,
        original_length: int
    ) -> float:
        """
        Get prediction using only the kept tokens (rationale/sufficient tokens).
        
        Args:
            kept_indices: Indices of tokens to keep
            hidden_states: Stored hidden states from original text (1, seq_len, hidden_dim)
            original_length: Original sequence length (actual, not padded)
            
        Returns:
            Model output probability for kept tokens only
        """
        # Extract only the actual sequence (remove padding)
        hidden_states = hidden_states.to(self.device)
        actual_hidden = hidden_states[0, :original_length, :]  # (original_length, hidden_dim)
        
        # Create mask for kept tokens (True = keep, False = remove)
        kept_mask = torch.zeros(original_length, dtype=torch.bool)
        for idx in kept_indices:
            kept_mask[idx] = True
        
        # Extract hidden states for kept tokens
        kept_hidden = actual_hidden[kept_mask, :].unsqueeze(0)  # (1, kept_len, hidden_dim)
        
        # Create attention mask for kept tokens
        kept_length = kept_hidden.shape[1]
        mask = torch.ones(1, kept_length, dtype=torch.bool, device=self.device)
        
        # Recompute attention on kept hidden states
        with torch.no_grad():
            context, _ = self.model.attention(kept_hidden, mask)
            prediction = self.model.classifier(context)
        
        return prediction.cpu().item()
    
    def compute_sufficiency(
        self,
        text: str,
        top_k: int = 5
    ) -> Dict:
        """
        Compute sufficiency metric for a single review using stored hidden states.
        
        Sufficiency measures whether the top-k attended tokens are sufficient
        for the model to make its prediction. 
        
        Steps:
        1. Get baseline prediction, attention scores, and hidden states
        2. Identify top-k high-attention tokens (get their indices)
        3. Extract hidden states for only these top-k tokens
        4. Recompute attention on only these kept hidden states
        5. Calculate sufficiency = m(xi)_j - m(ri)_j
        
        Args:
            text: Input review
            top_k: Number of top attended tokens to keep
            
        Returns:
            Dictionary with results:
                - original_prediction: m(xi)_j
                - rationale_prediction: m(ri)_j
                - sufficiency: m(xi)_j - m(ri)_j
                - original_tokens: List of original tokens
                - attention_weights: Attention scores
                - rationale_tokens: Top-k kept tokens
                - kept_indices: Indices of kept tokens
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
        rationale_tokens = [tokens[i] for i in top_k_indices]
        rationale_scores = attention_weights[top_k_indices]
        
        # Step 3 & 4: Keep only top-k and get new prediction using stored hidden states
        rationale_pred = self.get_kept_prediction(
            kept_indices=list(top_k_indices),
            hidden_states=hidden_states,
            original_length=original_length
        )
        
        # Step 5: Calculate sufficiency
        sufficiency = original_pred - rationale_pred
        
        return {
            "original_text": text,
            "original_prediction": original_pred,
            "rationale_prediction": rationale_pred,
            "sufficiency": sufficiency,
            "top_k": top_k,
            "original_tokens": tokens,
            "attention_weights": attention_weights.tolist(),
            "rationale_tokens": rationale_tokens,
            "rationale_scores": rationale_scores.tolist(),
            "kept_indices": list(map(int, top_k_indices))
        }
    
    def compute_multiple_k(
        self,
        text: str,
        k_values: List[int] = None
    ) -> Dict:
        """
        Compute sufficiency for multiple k values.
        
        Args:
            text: Input review
            k_values: List of k values to test
            
        Returns:
            Dictionary with results for each k:
                - results_by_k: Dict mapping k -> sufficiency results
                - original_text: Input text
                - original_tokens: Tokenized input
                - attention_weights: Full attention weight distribution
        """
        if k_values is None:
            k_values = ExperimentConfig.TOP_K_VALUES
        
        # Get baseline once
        baseline = self.get_baseline_prediction(text)
        original_pred = baseline["prediction"]
        attention_weights = baseline["attention_weights"]
        tokens = baseline["tokens"]
        hidden_states = baseline["hidden_states"]
        original_length = baseline["length"]
        
        results_by_k = {}
        
        for k in k_values:
            # Ensure k doesn't exceed sequence length
            k_actual = min(k, original_length)
            
            # Get indices of top-k
            top_k_indices = np.argsort(attention_weights)[-k_actual:]
            rationale_tokens = [tokens[i] for i in top_k_indices]
            rationale_scores = attention_weights[top_k_indices]
            
            # Get prediction with only top-k tokens
            rationale_pred = self.get_kept_prediction(
                kept_indices=list(top_k_indices),
                hidden_states=hidden_states,
                original_length=original_length
            )
            
            # Calculate sufficiency
            sufficiency = original_pred - rationale_pred
            
            results_by_k[k] = {
                "k": k,
                "original_prediction": original_pred,
                "rationale_prediction": rationale_pred,
                "sufficiency": sufficiency,
                "rationale_tokens": rationale_tokens,
                "rationale_scores": rationale_scores.tolist(),
                "kept_indices": list(map(int, top_k_indices))
            }
        
        return {
            "original_text": text,
            "original_tokens": tokens,
            "attention_weights": attention_weights.tolist(),
            "results_by_k": results_by_k
        }
