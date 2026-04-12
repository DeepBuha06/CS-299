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
    from experiment_2.adversarial_attack import run_adversarial_experiment, AdversarialAttentionAttack, compute_attention_difference
    from experiment_2.visualization import AdversarialVisualizer
    from experiment_2.comparison import AttentionComparator


def load_model():
    """Load the BiLSTM model and vocabulary."""
    project_root = Path(__file__).parent.parent
    
    # load vocabulary
    with open(project_root / 'vocab.json', 'r') as f:
        vocab = json.load(f)
    
    # load model
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
    
    return bilstm, vocab


def run_adversarial_on_text(model, text, vocab, model_type='bilstm'):
    
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
    
    viz = AdversarialVisualizer()
    
    visualizations = viz.generate_full_visualization(
        tokens=result['tokens'],
        original_attention=result['original_attention'],
        adversarial_attention=result['adversarial_attention'],
        metrics=result['difference_metrics']
    )
    
    details = AttentionComparator.generate_comparison_report(
        tokens=result['tokens'],
        original_attention=result['original_attention'],
        adversarial_attention=result['adversarial_attention'],
        original_prediction=result['original_prediction'],
        adversarial_prediction=result['adversarial_prediction']
    )
    
    return visualizations, details


def main():
    
    bilstm_model, vocab = load_model()
    
    test_texts = [
        ("Positive Review", "This movie was absolutely fantastic! Great acting and plot."),
        ("Negative Review", "Terrible movie! Complete waste of time and money."),
        ("Mixed Review", "The movie had good acting but the plot was boring.")
    ]
    
    for title, text in test_texts:
        
        result = run_adversarial_on_text(bilstm_model, text, vocab, 'bilstm')
        visualizations, details = visualize_results(result)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
