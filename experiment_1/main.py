"""
Experiment 1: Attention vs Feature Importance Correlation
Paper Section 4.1 — Algorithm 1

Tests whether attention weights correlate with gradient-based and
leave-one-out measures of feature importance using Kendall's τ.
"""

import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import AttentionClassifier
from config import Config

try:
    from feature_importance import FeatureImportanceAnalyzer, run_experiment_1
except ImportError:
    from experiment_1.feature_importance import FeatureImportanceAnalyzer, run_experiment_1


def load_model():
    """Load the BiLSTM model and vocabulary."""
    project_root = Path(__file__).parent.parent
    
    with open(project_root / 'vocab.json', 'r') as f:
        vocab = json.load(f)
    
    model = AttentionClassifier(
        vocab_size=Config.VOCAB_SIZE + 2,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        attention_dim=Config.ATTENTION_DIM,
        num_classes=Config.NUM_CLASSES,
        num_layers=Config.NUM_LAYERS,
        bidirectional=Config.BIDIRECTIONAL,
        attention_type=Config.ATTENTION_TYPE,
        encoder_dropout=Config.ENCODER_DROPOUT,
        classifier_dropout=Config.CLASSIFIER_DROPOUT,
        padding_idx=Config.PAD_IDX
    )
    
    model_path = project_root / 'checkpoints' / 'bilstm_model.pt'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("BiLSTM model loaded!")
    return model, vocab


def main():
    print("=" * 70)
    print("EXPERIMENT 1: ATTENTION vs FEATURE IMPORTANCE CORRELATION")
    print("Paper Section 4.1 — Algorithm 1")
    print("=" * 70)
    print()
    print("This experiment tests whether attention weights correlate with")
    print("gradient-based and leave-one-out measures of feature importance.")
    print("Low correlation → attention is NOT a reliable explanation.")
    print()
    
    model, vocab = load_model()
    
    test_texts = [
        "This movie was absolutely fantastic! Great acting and plot.",
        "Terrible movie! Complete waste of time and money.",
        "The movie had good acting but the plot was boring and predictable.",
        "An absolutely wonderful film with brilliant performances throughout.",
        "I hated every minute of this awful film. Never watching again.",
        "The cinematography was beautiful but the story lacked depth.",
        "One of the best movies I have ever seen truly remarkable work.",
        "Disappointing sequel that fails to capture the magic of the original.",
    ]
    
    print("Running analysis on", len(test_texts), "texts...\n")
    
    results = run_experiment_1(model, test_texts, vocab, device='cpu')
    
    # Print per-text results
    print("=" * 70)
    print("PER-TEXT RESULTS")
    print("=" * 70)
    
    for i, result in enumerate(results['individual_results']):
        print(f"\n--- Text {i+1} ---")
        print(f"Text: {result['text'][:80]}...")
        print(f"Prediction: {result['prediction']:.4f} ({'Positive' if result['prediction'] > 0.5 else 'Negative'})")
        print(f"τ_gradient: {result['correlations']['tau_gradient']:.4f} (p={result['correlations']['tau_gradient_pvalue']:.4f})")
        print(f"τ_loo:      {result['correlations']['tau_loo']:.4f} (p={result['correlations']['tau_loo_pvalue']:.4f})")
        
        # Show top 5 tokens by each measure
        sorted_by_attn = sorted(result['per_token_data'], key=lambda x: x['attention'], reverse=True)[:5]
        sorted_by_grad = sorted(result['per_token_data'], key=lambda x: x['gradient_importance'], reverse=True)[:5]
        sorted_by_loo = sorted(result['per_token_data'], key=lambda x: x['loo_importance'], reverse=True)[:5]
        
        print(f"\n  Top 5 by Attention:   {', '.join(d['token'] for d in sorted_by_attn)}")
        print(f"  Top 5 by Gradient:    {', '.join(d['token'] for d in sorted_by_grad)}")
        print(f"  Top 5 by LOO:         {', '.join(d['token'] for d in sorted_by_loo)}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (Paper Table 2 format)")
    print("=" * 70)
    summary = results['summary']
    print(f"\n  Gradient correlation (τ_g):  {summary['mean_tau_gradient']:.3f} ± {summary['std_tau_gradient']:.3f}")
    print(f"  LOO correlation (τ_loo):     {summary['mean_tau_loo']:.3f} ± {summary['std_tau_loo']:.3f}")
    print(f"  Fraction significant (grad): {summary['sig_frac_gradient']:.2f}")
    print(f"  Fraction significant (LOO):  {summary['sig_frac_loo']:.2f}")
    
    print(f"\n  Paper reports for IMDB BiLSTM:")
    print(f"    τ_g = 0.37 ± 0.08, τ_loo = 0.30 ± 0.07")
    
    if summary['mean_tau_gradient'] < 0.5:
        print(f"\n  → CONCLUSION: Attention weights show only WEAK correlation")
        print(f"    with feature importance measures, confirming the paper's finding")
        print(f"    that attention is NOT a faithful explanation.")
    else:
        print(f"\n  → Attention shows moderate-to-strong correlation with feature importance.")
    
    print("\n" + "=" * 70)
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
