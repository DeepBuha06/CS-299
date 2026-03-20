"""
Experiment 1: Attention vs Feature Importance Correlation (Paper Section 4.1)

Implements Algorithm 1 from "Attention is not Explanation" (Jain & Wallace, 2019).
Computes gradient-based and leave-one-out feature importance measures,
then correlates them with attention weights using Kendall's τ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import kendalltau
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import re


class FeatureImportanceAnalyzer:
    """
    Implements Algorithm 1 from the paper.
    Computes gradient and LOO importance, then correlates with attention.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def get_attention_and_prediction(
        self, 
        token_ids: torch.Tensor, 
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get attention weights and prediction from the model."""
        with torch.no_grad():
            predictions, attention_weights = self.model(token_ids, lengths, return_attention=True)
        return attention_weights.squeeze(0), predictions.squeeze()
    
    def compute_gradient_importance(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient-based feature importance (Algorithm 1, line 3).
        
        For each token t, importance = ||∂ŷ/∂embedding_t||₂
        
        We disconnect the gradient from flowing through the attention layer
        to measure "how important is this input to the output, given fixed attention."
        """
        self.model.eval()
        
        # We need gradients for this computation
        # Forward through model manually to control gradient flow
        batch_size, seq_len = token_ids.shape
        
        # Create mask
        mask = None
        if lengths is not None:
            range_tensor = torch.arange(seq_len, device=token_ids.device)
            range_tensor = range_tensor.unsqueeze(0).expand(batch_size, -1)
            mask = range_tensor < lengths.unsqueeze(1)
        
        embeddings = self.model.embedding(token_ids)  # (batch, seq_len, embed_dim)
        embeddings = embeddings.detach().requires_grad_(True)
        
        # Forward through encoder
        hidden_states, _ = self.model.encoder(embeddings, lengths)
        
        # Get attention weights but DETACH them from the graph
        # Paper: "we disconnect the computation graph at the attention module
        # so that the gradient does not flow through this layer"
        context_vector, attention_weights = self.model.attention(hidden_states, mask)
        
        # Detach attention and recompute context manually
        detached_attention = attention_weights.detach()  # Stop gradient here
        
        # Recompute context with detached attention: c = Σ α_t * h_t
        context_recomputed = torch.bmm(
            detached_attention.unsqueeze(1),  # (batch, 1, seq_len)
            hidden_states                      # (batch, seq_len, hidden_dim)
        ).squeeze(1)                          # (batch, hidden_dim)
        
        # Forward through classifier
        prediction = self.model.classifier(context_recomputed)
        
        # Backpropagate to get gradients w.r.t. embeddings
        prediction.sum().backward()
        
        # Gradient importance = L2 norm of gradient per token
        gradient_importance = embeddings.grad.norm(dim=-1).squeeze(0)  # (seq_len,)
        
        return gradient_importance.detach()
    
    def compute_loo_importance(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        pad_idx: int = 0
    ) -> torch.Tensor:
        """
        Compute Leave-One-Out importance (Algorithm 1, lines 4-5).
        
        For each token t:
            Δŷ_t = TVD(ŷ(x₋ₜ), ŷ(x))
        
        Where x₋ₜ is input with token t replaced by padding.
        TVD = Total Variation Distance = |ŷ(x₋ₜ) - ŷ(x)| for binary case.
        """
        seq_length = int(lengths[0].item())
        
        # Get original prediction
        with torch.no_grad():
            original_pred, _ = self.model(token_ids, lengths, return_attention=True)
            original_pred = original_pred.squeeze()
        
        loo_importance = torch.zeros(token_ids.shape[1])
        
        for t in range(seq_length):
            # Create masked input: replace token t with PAD
            masked_ids = token_ids.clone()
            masked_ids[0, t] = pad_idx
            
            with torch.no_grad():
                masked_pred, _ = self.model(masked_ids, lengths, return_attention=True)
                masked_pred = masked_pred.squeeze()
            
            # TVD for binary: |p_masked - p_original|
            loo_importance[t] = torch.abs(masked_pred - original_pred).item()
        
        return loo_importance
    
    def compute_correlations(
        self,
        attention_weights: torch.Tensor,
        gradient_importance: torch.Tensor,
        loo_importance: torch.Tensor
    ) -> Dict:
        """
        Compute Kendall's τ correlations (Algorithm 1, lines 3 and 5).
        
        τ_g   = Kendall-τ(α, g)    — attention vs gradients
        τ_loo = Kendall-τ(α, Δŷ)   — attention vs LOO
        """
        attn = attention_weights.detach().cpu().numpy()
        grad = gradient_importance.detach().cpu().numpy()
        loo = loo_importance.detach().cpu().numpy()

        def _kendall_tau_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
            # Need >=2 points and non-constant arrays for meaningful tau.
            if len(x) < 2 or len(y) < 2:
                return 0.0, 1.0
            if np.allclose(x, x[0]) or np.allclose(y, y[0]):
                return 0.0, 1.0

            tau, pval = kendalltau(x, y, variant='b', method='auto')
            if np.isnan(tau) or np.isnan(pval):
                return 0.0, 1.0
            return float(tau), float(pval)

        tau_g, p_g = _kendall_tau_safe(attn, grad)
        tau_loo, p_loo = _kendall_tau_safe(attn, loo)
        
        return {
            'tau_gradient': tau_g,
            'tau_gradient_pvalue': p_g,
            'tau_loo': tau_loo,
            'tau_loo_pvalue': p_loo,
            'gradient_significant': bool(p_g < 0.05),
            'loo_significant': bool(p_loo < 0.05)
        }
    
    def analyze_text(
        self,
        text: str,
        vocab: Dict,
        max_length: int = 256,
        pad_idx: int = 0
    ) -> Dict:
        """
        Run the full Algorithm 1 analysis on a single text.
        Returns attention, gradient importance, LOO importance, and correlations.
        """
        # Tokenize
        tokens = self._tokenize(text)[:max_length]
        
        # Convert to tensor
        ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
        length = len(ids)
        
        if len(ids) < max_length:
            ids.extend([pad_idx] * (max_length - len(ids)))
        
        token_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        lengths = torch.tensor([length]).to(self.device)
        
        # Step 1: Get attention weights and prediction
        attention_weights, prediction = self.get_attention_and_prediction(token_ids, lengths)
        
        # Step 2: Compute gradient importance
        gradient_importance = self.compute_gradient_importance(token_ids, lengths)
        
        # Step 3: Compute LOO importance
        loo_importance = self.compute_loo_importance(token_ids, lengths, pad_idx)
        
        # Keep only visible (non-padding) tokens and normalize each weight vector
        # over this filtered set so the webapp receives ready-to-render values.
        valid_indices = []
        for i in range(length):
            token = tokens[i].strip()
            if not token:
                continue
            if token.lower() in {'<pad>', '[pad]', 'pad'}:
                continue
            valid_indices.append(i)

        if valid_indices:
            idx_tensor = torch.tensor(valid_indices, device=attention_weights.device)
            filtered_tokens = [tokens[i] for i in valid_indices]
            filtered_attention = attention_weights.index_select(0, idx_tensor)
            filtered_gradient = gradient_importance.index_select(0, idx_tensor)
            filtered_loo = loo_importance.index_select(0, idx_tensor)
        else:
            filtered_tokens = []
            filtered_attention = torch.empty(0, device=attention_weights.device)
            filtered_gradient = torch.empty(0, device=gradient_importance.device)
            filtered_loo = torch.empty(0, device=loo_importance.device)

        def _normalize(weights: torch.Tensor) -> torch.Tensor:
            total = weights.sum()
            if weights.numel() == 0 or total.item() <= 0:
                return torch.zeros_like(weights)
            return weights / total

        normalized_attention = _normalize(filtered_attention)
        normalized_gradient = _normalize(filtered_gradient)
        normalized_loo = _normalize(filtered_loo)

        # Step 4: Compute correlations on the same filtered vectors sent to webapp.
        correlations = self.compute_correlations(
            filtered_attention,
            filtered_gradient,
            filtered_loo
        )
        
        # Per-token data for visualization
        per_token_data = []
        for i in range(len(filtered_tokens)):
            per_token_data.append({
                'token': filtered_tokens[i],
                'attention': float(normalized_attention[i]),
                'gradient_importance': float(normalized_gradient[i]),
                'loo_importance': float(normalized_loo[i])
            })
        
        return {
            'text': text,
            'tokens': filtered_tokens,
            'prediction': float(prediction),
            'attention': normalized_attention.detach().cpu().tolist(),
            'gradient_importance': normalized_gradient.detach().cpu().tolist(),
            'loo_importance': normalized_loo.detach().cpu().tolist(),
            'correlations': correlations,
            'per_token_data': per_token_data,
            'num_tokens': len(filtered_tokens)
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        words = re.findall(r'\b[a-z]+\b', text)
        return words


def run_experiment_1(
    model: nn.Module,
    texts: List[str],
    vocab: Dict,
    device: str = 'cpu',
    max_length: int = 256
) -> Dict:
    """
    Run Experiment 1 on multiple texts and aggregate statistics.
    """
    analyzer = FeatureImportanceAnalyzer(model, device)
    
    results = []
    all_tau_g = []
    all_tau_loo = []
    
    for text in texts:
        result = analyzer.analyze_text(text, vocab, max_length)
        results.append(result)
        all_tau_g.append(result['correlations']['tau_gradient'])
        all_tau_loo.append(result['correlations']['tau_loo'])
    
    summary = {
        'num_texts': len(texts),
        'mean_tau_gradient': float(np.mean(all_tau_g)),
        'std_tau_gradient': float(np.std(all_tau_g)),
        'mean_tau_loo': float(np.mean(all_tau_loo)),
        'std_tau_loo': float(np.std(all_tau_loo)),
        'sig_frac_gradient': sum(1 for r in results if r['correlations']['gradient_significant']) / len(results),
        'sig_frac_loo': sum(1 for r in results if r['correlations']['loo_significant']) / len(results),
    }
    
    return {
        'individual_results': results,
        'summary': summary
    }
