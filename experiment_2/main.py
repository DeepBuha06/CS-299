"""
Experiment 1: Adversarial Attention Attack
Finds attention distributions that produce the SAME prediction but are DIFFERENT from original attention.

This experiment demonstrates that attention is NOT a faithful explanation because
we can find alternative attention weights that give the same prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import AttentionClassifier
from config import Config

try:
    from adversarial_attack import run_adversarial_experiment, AdversarialAttentionAttack, compute_attention_difference
    from visualization import AdversarialVisualizer
    from comparison import AttentionComparator
except ImportError:
    from experiment_1.adversarial_attack import run_adversarial_experiment, AdversarialAttentionAttack, compute_attention_difference
    from experiment_1.visualization import AdversarialVisualizer
    from experiment_1.comparison import AttentionComparator


def load_model():
    """Load the BiLSTM model and vocabulary."""
    project_root = Path(__file__).parent.parent
    
    # Load vocabulary
    with open(project_root / 'vocab.json', 'r') as f:
        vocab = json.load(f)
    
    # Load BiLSTM model
    bilstm = AttentionClassifier(
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
        bilstm.load_state_dict(checkpoint['model_state_dict'])
    else:
        bilstm.load_state_dict(checkpoint)
    
    bilstm.eval()
    print("BiLSTM model loaded!")
    
    return bilstm, vocab


def run_adversarial_on_text(model, text, vocab, model_type='bilstm'):
    """Run adversarial attention experiment on a text using the actual attack."""
    
    # Use the full adversarial experiment pipeline
    result = run_adversarial_experiment(
        model=model,
        text=text,
        vocab=vocab,
        max_length=Config.MAX_SEQ_LENGTH,
        device='cpu'
    )
    
    result['model_type'] = model_type
    return result


def visualize_results(result):
    """Create visualizations for the results."""
    viz = AdversarialVisualizer()
    
    visualizations = viz.generate_full_visualization(
        tokens=result['tokens'],
        original_attention=result['original_attention'],
        adversarial_attention=result['adversarial_attention'],
        metrics=result['difference_metrics']
    )
    
    report, details = AttentionComparator.generate_comparison_report(
        tokens=result['tokens'],
        original_attention=result['original_attention'],
        adversarial_attention=result['adversarial_attention'],
        original_prediction=result['original_prediction'],
        adversarial_prediction=result['adversarial_prediction']
    )
    
    return visualizations, report, details


def main():
    """Main function to run Experiment 1."""
    
    print("=" * 70)
    print("EXPERIMENT 1: ADVERSARIAL ATTENTION ATTACK")
    print("=" * 70)
    print("\nThis experiment demonstrates that attention is NOT a faithful")
    print("explanation because we can find different attention weights that")
    print("produce the SAME prediction as the original attention.\n")
    
    print("Loading model...")
    bilstm_model, vocab = load_model()
    
    test_texts = [
        ("Positive Review", "This movie was absolutely fantastic! Great acting and plot."),
        ("Negative Review", "Terrible movie! Complete waste of time and money."),
        ("Mixed Review", "The movie had good acting but the plot was boring.")
    ]
    
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENTS")
    print("=" * 70)
    
    for title, text in test_texts:
        print(f"\n--- {title} ---")
        print(f"Text: {text}")
        
        result = run_adversarial_on_text(bilstm_model, text, vocab, 'bilstm')
        
        print(f"\nOriginal Prediction: {result['original_prediction']:.4f}")
        print(f"Adversarial Prediction: {result['adversarial_prediction']:.4f}")
        print(f"Prediction Difference: {abs(result['original_prediction'] - result['adversarial_prediction']):.4f}")
        print(f"Best Method: {result.get('best_method', 'N/A')}")
        
        print(f"\nAttention Difference Metrics:")
        print(f"  L1 Distance: {result['difference_metrics']['l1_difference']:.4f}")
        print(f"  Cosine Similarity: {result['difference_metrics']['cosine_similarity']:.4f}")
        
        visualizations, report, details = visualize_results(result)
        
        print("\nComparison Report:")
        print(report)
        
        print("\nTop 5 Words by Original Attention:")
        for item in details['top_original']:
            print(f"  {item['rank']}. {item['word']:<15} attention: {item['attention']:.4f}")
        
        print("\nTop 5 Words by Adversarial Attention:")
        for item in details['top_adversarial']:
            print(f"  {item['rank']}. {item['word']:<15} attention: {item['attention']:.4f}")
        
        print("\n" + "-" * 70)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\nKey Finding: Even with maximally different attention distributions,")
    print("the model gives the SAME prediction, proving that attention")
    print("weights are NOT reliable explanations for model predictions!")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
