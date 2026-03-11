"""
Comparison utilities for analyzing original vs adversarial attention.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import json


class AttentionComparator:
    """Compare original and adversarial attention distributions."""
    
    @staticmethod
    def compute_correlation(
        original: torch.Tensor,
        adversarial: torch.Tensor
    ) -> float:
        """Compute Pearson correlation between attention distributions."""
        orig_np = original.cpu().numpy() if isinstance(original, torch.Tensor) else np.array(original)
        adv_np = adversarial.cpu().numpy() if isinstance(adversarial, torch.Tensor) else np.array(adversarial)
        
        correlation = np.corrcoef(orig_np, adv_np)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def compute_kl_divergence(
        original: torch.Tensor,
        adversarial: torch.Tensor
    ) -> float:
        """Compute KL divergence from original to adversarial."""
        orig_np = original.cpu().numpy() if isinstance(original, torch.Tensor) else np.array(original)
        adv_np = adversarial.cpu().numpy() if isinstance(adversarial, torch.Tensor) else np.array(adversarial)
        
        orig_np = np.clip(orig_np, 1e-10, 1.0)
        adv_np = np.clip(adv_np, 1e-10, 1.0)
        
        kl = np.sum(orig_np * np.log(orig_np / adv_np))
        return float(kl)
    
    @staticmethod
    def compute_js_divergence(
        original: torch.Tensor,
        adversarial: torch.Tensor
    ) -> float:
        """Compute Jensen-Shannon divergence."""
        orig_np = original.cpu().numpy() if isinstance(original, torch.Tensor) else np.array(original)
        adv_np = adversarial.cpu().numpy() if isinstance(adversarial, torch.Tensor) else np.array(adversarial)
        
        orig_np = np.clip(orig_np, 1e-10, 1.0)
        adv_np = np.clip(adv_np, 1e-10, 1.0)
        
        m = 0.5 * (orig_np + adv_np)
        
        kl_orig_m = np.sum(orig_np * np.log(orig_np / m))
        kl_adv_m = np.sum(adv_np * np.log(adv_np / m))
        
        js = 0.5 * (kl_orig_m + kl_adv_m)
        return float(js)
    
    @staticmethod
    def get_top_attention_words(
        tokens: List[str],
        attention: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        """Get top-k words with highest attention."""
        combined = list(zip(tokens, attention))
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {'word': word, 'attention': float(attn), 'rank': i + 1}
            for i, (word, attn) in enumerate(combined[:top_k])
        ]
    
    @staticmethod
    def compute_attention_shift(
        tokens: List[str],
        original: List[float],
        adversarial: List[float]
    ) -> Dict:
        """Analyze how attention shifts between original and adversarial."""
        
        orig_sorted = sorted(range(len(original)), key=lambda i: original[i], reverse=True)
        adv_sorted = sorted(range(len(adversarial)), key=lambda i: adversarial[i], reverse=True)
        
        rank_changes = {}
        for i, idx in enumerate(orig_sorted):
            orig_rank = i
            adv_rank = adv_sorted.index(idx)
            rank_changes[tokens[idx]] = {
                'original_rank': orig_rank + 1,
                'adversarial_rank': adv_rank + 1,
                'rank_change': orig_rank - adv_rank
            }
        
        return rank_changes
    
    @staticmethod
    def generate_comparison_report(
        tokens: List[str],
        original_attention: List[float],
        adversarial_attention: List[float],
        original_prediction: float,
        adversarial_prediction: float
    ) -> str:
        """Generate a comprehensive comparison report."""
        
        orig_tensor = torch.tensor(original_attention)
        adv_tensor = torch.tensor(adversarial_attention)
        
        metrics = {
            'l1_difference': float(torch.abs(orig_tensor - adv_tensor).sum()),
            'l2_difference': float(torch.sqrt(((orig_tensor - adv_tensor) ** 2).sum())),
            'max_difference': float(torch.abs(orig_tensor - adv_tensor).max()),
            'mean_difference': float(torch.abs(orig_tensor - adv_tensor).mean()),
            'pearson_correlation': AttentionComparator.compute_correlation(orig_tensor, adv_tensor),
            'kl_divergence': AttentionComparator.compute_kl_divergence(orig_tensor, adv_tensor),
            'js_divergence': AttentionComparator.compute_js_divergence(orig_tensor, adv_tensor)
        }
        
        top_original = AttentionComparator.get_top_attention_words(tokens, original_attention, 5)
        top_adversarial = AttentionComparator.get_top_attention_words(tokens, adversarial_attention, 5)
        
        rank_changes = AttentionComparator.compute_attention_shift(
            tokens, original_attention, adversarial_attention
        )
        
        report = f"""
================================================================================
                    ADVERSARIAL ATTACK COMPARISON REPORT
================================================================================

PREDICTIONS:
  Original Prediction:     {original_prediction:.4f}
  Adversarial Prediction: {adversarial_prediction:.4f}
  Prediction Difference:  {abs(original_prediction - adversarial_prediction):.4f}

ATTENTION DIFFERENCE METRICS:
  L1 Distance:             {metrics['l1_difference']:.4f}
  L2 Distance:            {metrics['l2_difference']:.4f}
  Maximum Difference:     {metrics['max_difference']:.4f}
  Mean Difference:        {metrics['mean_difference']:.4f}
  Pearson Correlation:    {metrics['pearson_correlation']:.4f}
  KL Divergence:          {metrics['kl_divergence']:.4f}
  JS Divergence:          {metrics['js_divergence']:.4f}

TOP 5 WORDS (Original Attention):
"""
        
        for item in top_original:
            report += f"  {item['rank']}. {item['word']:<15} attention: {item['attention']:.4f}\n"
        
        report += "\nTOP 5 WORDS (Adversarial Attention):\n"
        for item in top_adversarial:
            report += f"  {item['rank']}. {item['word']:<15} attention: {item['attention']:.4f}\n"
        
        report += "\nKEY FINDINGS:\n"
        
        if metrics['pearson_correlation'] < 0.5:
            report += "  - Low correlation indicates significantly different attention patterns\n"
        else:
            report += "  - High correlation indicates similar attention patterns\n"
        
        if abs(original_prediction - adversarial_prediction) < 0.05:
            report += "  - SAME predictions despite different attention (proves attention is not explanation!)\n"
        
        report += "\n================================================================================\n"
        
        return report, {
            'metrics': metrics,
            'top_original': top_original,
            'top_adversarial': top_adversarial,
            'rank_changes': rank_changes,
            'predictions': {
                'original': original_prediction,
                'adversarial': adversarial_prediction,
                'difference': abs(original_prediction - adversarial_prediction)
            }
        }


def batch_compare_attentions(
    results: List[Dict]
) -> Dict:
    """Compare multiple attention pairs and compute aggregate statistics."""
    
    all_l1 = []
    all_correlations = []
    all_prediction_diffs = []
    
    for result in results:
        orig = torch.tensor(result['original_attention'])
        adv = torch.tensor(result['adversarial_attention'])
        
        l1 = float(torch.abs(orig - adv).sum())
        corr = AttentionComparator.compute_correlation(orig, adv)
        pred_diff = abs(result['original_prediction'] - result['adversarial_prediction'])
        
        all_l1.append(l1)
        all_correlations.append(corr)
        all_prediction_diffs.append(pred_diff)
    
    return {
        'num_samples': len(results),
        'average_l1_difference': np.mean(all_l1),
        'std_l1_difference': np.std(all_l1),
        'average_correlation': np.mean(all_correlations),
        'std_correlation': np.std(all_correlations),
        'average_prediction_diff': np.mean(all_prediction_diffs),
        'same_prediction_count': sum(1 for d in all_prediction_diffs if d < 0.05),
        'same_prediction_rate': sum(1 for d in all_prediction_diffs if d < 0.05) / len(results)
    }


if __name__ == '__main__':
    tokens = ['the', 'movie', 'was', 'really', 'great', 'and', 'fantastic']
    original = [0.05, 0.15, 0.08, 0.12, 0.35, 0.10, 0.15]
    adversarial = [0.14, 0.14, 0.14, 0.14, 0.15, 0.14, 0.15]
    
    report, details = AttentionComparator.generate_comparison_report(
        tokens, original, adversarial, 0.85, 0.84
    )
    
    print(report)
    print("\nMetrics JSON:")
    print(json.dumps(details['metrics'], indent=2))
