"""
Webapp routes for Experiment 1: Attention vs Feature Importance Correlation.
Paper Section 4.1 — Algorithm 1
"""

import json
from pathlib import Path
from flask import Blueprint, request, jsonify

import torch

from models.model import AttentionClassifier
from config import Config
from experiment_1.feature_importance import FeatureImportanceAnalyzer

experiment1_bp = Blueprint('experiment1', __name__, url_prefix='/experiment1')

bilstm_model = None
vocab = None
initialized = False


def initialize_experiment1_models():
    """Initialize models for experiment 1."""
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
        print(f"Failed to initialize experiment 1 models: {e}")
        bilstm_model = None
        return False


@experiment1_bp.route('/analyze', methods=['POST'])
def analyze_correlation():
    """Run attention vs feature importance correlation analysis."""
    try:
        if not initialize_experiment1_models():
            return jsonify({'error': 'Model could not be loaded.'}), 500

        data = request.get_json()
        text = data.get('text', '')

        if not text.strip():
            return jsonify({'error': 'Please provide text for analysis.'}), 400

        analyzer = FeatureImportanceAnalyzer(bilstm_model, device='cpu')
        result = analyzer.analyze_text(text, vocab, max_length=Config.MAX_SEQ_LENGTH)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@experiment1_bp.route('/sample', methods=['GET'])
def get_sample_texts():
    """Get sample texts for demonstration."""
    samples = {
        'positive': "This movie was absolutely fantastic! Great acting and brilliant storyline.",
        'negative': "Terrible movie! Complete waste of time and money. Very disappointing.",
        'neutral': "The movie had interesting cinematography but the plot was somewhat predictable."
    }
    return jsonify(samples)
