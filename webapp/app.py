"""
Flask Web Application for Sentiment Analysis
Uses the trained BiLSTM Attention model to classify reviews as positive or negative.
"""

import sys
import json
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify
import torch

from models.model import AttentionClassifier
from config import Config

app = Flask(__name__)

# Global model and vocab
model = None
vocab = None


def tokenize(text):
    """Simple tokenizer: lowercase and extract words."""
    text = text.lower()
    # Remove HTML tags like <br />
    text = re.sub(r'<[^>]+>', ' ', text)
    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b[a-z]+\b', text)
    return words


def text_to_tensor(text, vocab, max_length=256):
    """Convert text to tensor of token IDs."""
    tokens = tokenize(text)
    
    # Convert to IDs
    ids = []
    for token in tokens[:max_length]:
        ids.append(vocab.get(token, Config.UNK_IDX))
    
    # Pad if necessary
    length = len(ids)
    if length < max_length:
        ids.extend([Config.PAD_IDX] * (max_length - length))
    
    return torch.tensor([ids]), torch.tensor([length])


def load_model():
    """Load the trained model and vocabulary."""
    global model, vocab
    
    project_root = Path(__file__).parent.parent
    
    # Load vocabulary
    vocab_path = project_root / "vocab.json"
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    print(f"Loaded vocabulary with {len(vocab)} words")
    
    # Create model with same config used during training
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
    
    # Load trained weights
    model_path = project_root / "checkpoints" / "best_model.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!")


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Analyze the sentiment of the given review."""
    try:
        data = request.get_json()
        review_text = data.get('review', '')
        
        if not review_text.strip():
            return jsonify({'error': 'Please enter a review'}), 400
        
        # Convert text to tensor
        token_ids, lengths = text_to_tensor(review_text, vocab, Config.MAX_SEQ_LENGTH)
        
        # Get prediction and attention weights
        with torch.no_grad():
            predictions, attention_weights = model(token_ids, lengths, return_attention=True)
            probability = predictions.item()
        
        # Determine sentiment
        is_positive = probability >= 0.5
        sentiment = 'positive' if is_positive else 'negative'
        confidence = probability if is_positive else (1 - probability)
        
        # Get top attention words for visualization
        tokens = tokenize(review_text)[:Config.MAX_SEQ_LENGTH]
        attention_data = []
        
        if attention_weights is not None:
            weights = attention_weights[0][:len(tokens)].tolist()
            for token, weight in zip(tokens, weights):
                attention_data.append({
                    'word': token,
                    'weight': weight
                })
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence * 100, 1),
            'probability': round(probability * 100, 1),
            'attention': attention_data
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/metrics')
def metrics():
    """Return model metrics."""
    project_root = Path(__file__).parent.parent
    metrics_path = project_root / "checkpoints" / "metrics.json"
    
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    return jsonify(metrics_data)


if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
