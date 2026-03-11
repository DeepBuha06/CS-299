"""
Visualization utilities for adversarial attention experiment.
Creates graphs and charts to compare original vs adversarial attention.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from pathlib import Path
import base64
from io import BytesIO


class AdversarialVisualizer:
    """Create visualizations for adversarial attention experiment."""
    
    def __init__(self, figsize=(12, 8), style='seaborn-v0_8-darkgrid'):
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
    
    def create_comparison_bar_chart(
        self,
        tokens: List[str],
        original_attention: List[float],
        adversarial_attention: List[float],
        top_n: int = 15
    ) -> str:
        """Create side-by-side bar chart comparing original and adversarial attention."""
        
        if len(tokens) > top_n:
            combined = list(zip(tokens, original_attention, adversarial_attention))
            combined.sort(key=lambda x: x[1], reverse=True)
            tokens = [c[0] for c in combined[:top_n]]
            original_attention = [c[1] for c in combined[:top_n]]
            adversarial_attention = [c[2] for c in combined[:top_n]]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(tokens))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, original_attention, width, 
                      label='Original Attention', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, adversarial_attention, width,
                      label='Adversarial Attention', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Tokens', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title('Original vs Adversarial Attention Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64
    
    def create_difference_heatmap(
        self,
        tokens: List[str],
        original_attention: List[float],
        adversarial_attention: List[float]
    ) -> str:
        """Create a heatmap showing attention difference."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        difference = [abs(o - a) for o, a in zip(original_attention, adversarial_attention)]
        
        tokens_to_show = tokens[:20] if len(tokens) > 20 else tokens
        difference_to_show = difference[:20] if len(difference) > 20 else difference
        
        colors = plt.cm.Reds(np.array(difference_to_show) / max(difference_to_show) if max(difference_to_show) > 0 else [0])
        
        ax.barh(range(len(tokens_to_show)), difference_to_show, color=colors)
        ax.set_yticks(range(len(tokens_to_show)))
        ax.set_yticklabels(tokens_to_show)
        ax.set_xlabel('|Original - Adversarial|', fontsize=12)
        ax.set_title('Attention Difference per Token', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64
    
    def create_scatter_comparison(
        self,
        tokens: List[str],
        original_attention: List[float],
        adversarial_attention: List[float]
    ) -> str:
        """Create scatter plot comparing original vs adversarial attention."""
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.scatter(original_attention, adversarial_attention, alpha=0.6, s=100, c='blue')
        
        max_val = max(max(original_attention), max(adversarial_attention))
        ax.plot([0, max_val], [0, max_val], 'r--', label='y=x (equal attention)')
        
        for i, token in enumerate(tokens[:10]):
            ax.annotate(token, (original_attention[i], adversarial_attention[i]),
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Original Attention', fontsize=12)
        ax.set_ylabel('Adversarial Attention', fontsize=12)
        ax.set_title('Original vs Adversarial Attention Scatter', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64
    
    def create_metrics_summary(
        self,
        metrics: Dict
    ) -> str:
        """Create a text-based summary of metrics."""
        
        summary = f"""
╔════════════════════════════════════════════════════════════╗
║           ADVERSARIAL ATTACK RESULTS SUMMARY               ║
╠════════════════════════════════════════════════════════════╣
║  L1 Difference:         {metrics.get('l1_difference', 0):.4f}                        ║
║  L2 Difference:        {metrics.get('l2_difference', 0):.4f}                        ║
║  Max Difference:       {metrics.get('max_difference', 0):.4f}                        ║
║  Mean Difference:      {metrics.get('mean_difference', 0):.4f}                        ║
║  Cosine Similarity:    {metrics.get('cosine_similarity', 0):.4f}                        ║
╚════════════════════════════════════════════════════════════╝
        """
        return summary.strip()
    
    def create_word_highlight_html(
        self,
        tokens: List[str],
        attention: List[float],
        attention_type: str = 'original'
    ) -> str:
        """Create HTML with word highlighting based on attention weights."""
        
        max_attention = max(attention) if max(attention) > 0 else 1.0
        
        html_parts = []
        for token, attn in zip(tokens, attention):
            normalized = attn / max_attention
            alpha = 0.3 + 0.7 * normalized
            
            color = '#ff6b6b' if attention_type == 'original' else '#4ecdc4'
            
            html_parts.append(
                f'<span style="background-color: {color}; opacity: {alpha:.2f}; '
                f'padding: 2px 4px; margin: 1px; border-radius: 3px;" '
                f'title="Attention: {attn:.4f}">{token}</span>'
            )
        
        return ''.join(html_parts)
    
    def generate_full_visualization(
        self,
        tokens: List[str],
        original_attention: List[float],
        adversarial_attention: List[float],
        metrics: Dict
    ) -> Dict:
        """Generate all visualizations and return as base64."""
        
        visualizations = {}
        
        visualizations['bar_chart'] = self.create_comparison_bar_chart(
            tokens, original_attention, adversarial_attention
        )
        
        visualizations['heatmap'] = self.create_difference_heatmap(
            tokens, original_attention, adversarial_attention
        )
        
        visualizations['scatter'] = self.create_scatter_comparison(
            tokens, original_attention, adversarial_attention
        )
        
        visualizations['summary'] = self.create_metrics_summary(metrics)
        
        visualizations['original_words'] = self.create_word_highlight_html(
            tokens, original_attention, 'original'
        )
        
        visualizations['adversarial_words'] = self.create_word_highlight_html(
            tokens, adversarial_attention, 'adversarial'
        )
        
        return visualizations


def create_sample_visualizations():
    """Create sample visualizations for demonstration."""
    
    tokens = ['the', 'movie', 'was', 'really', 'great', 'and', 'fantastic']
    original = [0.05, 0.15, 0.08, 0.12, 0.35, 0.10, 0.15]
    adversarial = [0.14, 0.14, 0.14, 0.14, 0.15, 0.14, 0.15]
    
    viz = AdversarialVisualizer()
    
    metrics = {
        'l1_difference': sum(abs(o - a) for o, a in zip(original, adversarial)),
        'l2_difference': (sum((o - a) ** 2 for o, a in zip(original, adversarial))) ** 0.5,
        'max_difference': max(abs(o - a) for o, a in zip(original, adversarial)),
        'mean_difference': sum(abs(o - a) for o, a in zip(original, adversarial)) / len(original),
        'cosine_similarity': sum(o * a for o, a in zip(original, adversarial)) / (
            (sum(o**2 for o in original) ** 0.5) * (sum(a**2 for a in adversarial) ** 0.5)
        )
    }
    
    return viz.generate_full_visualization(tokens, original, adversarial, metrics)


if __name__ == '__main__':
    results = create_sample_visualizations()
    print("Sample visualizations generated successfully!")
    print(results['summary'])
