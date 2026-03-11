"""
Flask routes for Experiment 1 - Adversarial Attention
"""

import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Blueprint, request, jsonify
import torch

from models.model import AttentionClassifier
from config import Config
from experiment_2.adversarial_attack import run_adversarial_experiment, compute_attention_difference
from experiment_2.visualization import AdversarialVisualizer
from experiment_2.comparison import AttentionComparator

experiment2_bp = Blueprint('experiment2', __name__, url_prefix='/experiment2')

bilstm_model = None
vocab = None
initialized = False


def initialize_experiment_models():
    """Initialize models for experiment."""
    global bilstm_model, vocab, initialized
    
    if initialized:
        return True
    
    project_root = Path(__file__).parent.parent
    
    try:
        with open(project_root / 'vocab.json', 'r') as f:
            vocab = json.load(f)
        
        bilstm_model = AttentionClassifier(
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
        if not model_path.exists():
            print(f"Model checkpoint not found at {model_path}")
            return False
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            bilstm_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            bilstm_model.load_state_dict(checkpoint)
        
        bilstm_model.eval()
        
        initialized = True
        print("Experiment 1 models initialized!")
        return True
    except Exception as e:
        print(f"Failed to initialize experiment models: {e}")
        bilstm_model = None
        return False


@experiment2_bp.route('/analyze', methods=['POST'])
def analyze_adversarial():
    """Run adversarial attention analysis on input text."""
    try:
        if not initialize_experiment_models():
            return jsonify({'error': 'Model could not be loaded. Make sure bilstm_model.pt exists in checkpoints/'}), 500
        
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'Please enter text'}), 400
        
        result = run_adversarial_experiment(
            model=bilstm_model,
            text=text,
            vocab=vocab,
            max_length=Config.MAX_SEQ_LENGTH,
            device='cpu'
        )
        
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
        
        response = {
            'text': result['text'],
            'tokens': result['tokens'],
            'original_attention': result['original_attention'],
            'adversarial_attention': result['adversarial_attention'],
            'original_prediction': float(result['original_prediction']),
            'adversarial_prediction': float(result['adversarial_prediction']),
            'prediction_difference': float(abs(result['original_prediction'] - result['adversarial_prediction'])),
            'best_method': result['best_method'],
            'attention_difference': float(result['attention_difference']),
            'difference_metrics': result['difference_metrics'],
            'top_original': details['top_original'],
            'top_adversarial': details['top_adversarial'],
            'visualizations': visualizations,
            'report': report
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@experiment2_bp.route('/sample', methods=['GET'])
def get_sample_texts():
    """Get sample texts for demonstration."""
    samples = {
        'positive': [
            "This movie was absolutely fantastic! Great acting and plot.",
            "I loved every minute of this film. Highly recommend!",
            "What an amazing movie! The best I've seen this year."
        ],
        'negative': [
            "Terrible movie! Complete waste of time and money.",
            "This is the worst film I've ever seen. Do not watch.",
            "Boring, predictable, and poorly made. Avoid this movie."
        ],
        'neutral': [
            "The movie was okay. Nothing special but not terrible either.",
            "Average film with some good parts and some bad parts."
        ]
    }
    return jsonify(samples)


@experiment2_bp.route('/info', methods=['GET'])
def get_experiment_info():
    """Get information about the experiment."""
    info = {
        'title': 'Experiment 1: Adversarial Attention Attack',
        'description': '''
            This experiment demonstrates that attention is NOT a faithful explanation
            for model predictions. We find alternative (adversarial) attention 
            distributions that produce the SAME prediction as the original attention
            but are maximally different in their attention weights.
        ''',
        'key_findings': [
            'Different attention patterns can yield the same prediction',
            'High attention words are not necessarily important for the prediction',
            'Attention weights alone cannot be trusted as explanations'
        ],
        'methods': [
            'Entropy Maximization (most uniform attention)',
            'Random Permutation',
            'Gradient-based Optimization'
        ]
    }
    return jsonify(info)


@experiment2_bp.route('/batch', methods=['POST'])
def run_batch_analysis():
    """Run analysis on multiple texts."""
    try:
        initialize_experiment_models()
        
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        results = []
        
        for text in texts:
            result = run_adversarial_experiment(
                model=bilstm_model,
                text=text,
                vocab=vocab,
                max_length=Config.MAX_SEQ_LENGTH,
                device='cpu'
            )
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'original_prediction': float(result['original_prediction']),
                'adversarial_prediction': float(result['adversarial_prediction']),
                'attention_difference': float(result['attention_difference']),
                'l1_difference': float(result['difference_metrics']['l1_difference']),
                'cosine_similarity': float(result['difference_metrics']['cosine_similarity'])
            })
        
        avg_l1 = sum(r['l1_difference'] for r in results) / len(results)
        avg_cosine = sum(r['cosine_similarity'] for r in results) / len(results)
        
        same_prediction_rate = sum(
            1 for r in results if abs(r['original_prediction'] - r['adversarial_prediction']) < 0.05
        ) / len(results)
        
        return jsonify({
            'individual_results': results,
            'summary': {
                'num_samples': len(results),
                'average_l1_difference': avg_l1,
                'average_cosine_similarity': avg_cosine,
                'same_prediction_rate': same_prediction_rate
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
