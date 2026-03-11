"""
Flask Web Application for Sentiment Analysis
Supports both BiLSTM Attention model and Transformer (DistilBERT) model.
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
from config_transformer import TransformerConfig
from models_transformer.model import TransformerClassifier
from models_transformer.dataset import get_tokenizer
from webapp.experiment_routes import experiment2_bp
from webapp.experiment1_routes import experiment1_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(experiment2_bp)
app.register_blueprint(experiment1_bp)

# Global model and vocab
bilstm_model = None
transformer_model = None
vocab = None
transformer_tokenizer = None


def tokenize(text):
    """Simple tokenizer: lowercase and extract words."""
    text = text.lower()
    # Remove HTML tags like <br />
    text = re.sub(r'<[^>]+>', ' ', text)
    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b[a-z]+\b', text)
    return words


def text_to_tensor(text, vocab, max_length=256):
    """Convert text to tensor of token IDs for BiLSTM model."""
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


def load_bilstm_model():
    """Load the trained BiLSTM model and vocabulary."""
    global bilstm_model, vocab
    
    project_root = Path(__file__).parent.parent
    
    # Load vocabulary
    vocab_path = project_root / "vocab.json"
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    print(f"Loaded vocabulary with {len(vocab)} words")
    
    # Create model with same config used during training
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
    
    # Load trained weights
    model_path = project_root / "checkpoints" / "bilstm_model.pt"
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            bilstm_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            bilstm_model.load_state_dict(checkpoint)
        
        bilstm_model.eval()
        print("BiLSTM model loaded successfully!")
        return True
    else:
        print(f"BiLSTM model not found at {model_path}")
        return False


def load_transformer_model():
    """Load the trained Transformer model and tokenizer."""
    global transformer_model, transformer_tokenizer
    
    project_root = Path(__file__).parent.parent
    
    # Load tokenizer
    transformer_tokenizer = get_tokenizer(TransformerConfig.MODEL_NAME)
    print("Loaded transformer tokenizer")
    
    # Create model
    transformer_model = TransformerClassifier(
        model_name=TransformerConfig.MODEL_NAME,
        num_labels=TransformerConfig.NUM_LABELS
    )
    
    # Load trained weights
    model_path = project_root / "checkpoints" / TransformerConfig.MODEL_CHECKPOINT
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            transformer_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            transformer_model.load_state_dict(checkpoint)
        
        transformer_model.eval()
        print("Transformer model loaded successfully!")
        return True
    else:
        print(f"Transformer model not found at {model_path}")
        return False


def predict_with_bilstm(text):
    """Make prediction using BiLSTM model."""
    # Convert text to tensor
    token_ids, lengths = text_to_tensor(text, vocab, Config.MAX_SEQ_LENGTH)
    
    # Get prediction and attention weights
    with torch.no_grad():
        predictions, attention_weights = bilstm_model(token_ids, lengths, return_attention=True)
        probability = predictions.item()
    
    # Determine sentiment
    is_positive = probability >= 0.5
    sentiment = 'positive' if is_positive else 'negative'
    confidence = probability if is_positive else (1 - probability)
    
    # Get top attention words for visualization
    tokens = tokenize(text)[:Config.MAX_SEQ_LENGTH]
    attention_data = []
    
    if attention_weights is not None:
        weights = attention_weights[0][:len(tokens)].tolist()
        for token, weight in zip(tokens, weights):
            attention_data.append({
                'word': token,
                'weight': weight
            })
    
    return {
        'sentiment': sentiment,
        'confidence': round(confidence * 100, 1),
        'probability': round(probability * 100, 1),
        'attention': attention_data,
        'model': 'BiLSTM + Attention'
    }


def predict_with_transformer(text):
    """Make prediction using Transformer model."""
    # Tokenize
    encoding = transformer_tokenizer(
        text,
        truncation=True,
        max_length=TransformerConfig.MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    # Get prediction
    with torch.no_grad():
        logits, attention = transformer_model(input_ids, attention_mask, return_attention=True)
        probabilities = torch.softmax(logits, dim=-1)
    
    # Get predicted class and confidence
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class].item()
    probability_positive = probabilities[0, 1].item()
    
    sentiment = 'positive' if predicted_class == 1 else 'negative'
    
    # Get tokens for visualization
    tokens = transformer_tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Filter out padding tokens
    actual_length = attention_mask[0].sum().item()
    tokens = tokens[:actual_length]
    
    # Build attention data
    attention_data = []
    
    # Punctuation and special tokens to filter out from attention visualization
    # These often get high attention but aren't semantically meaningful
    punctuation_tokens = {
        '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]',
        '.', ',', '!', '?', ';', ':', '"', "'", '-', '(', ')', '[', ']', '{', '}',
        '...', '--', '``', "''", '/', '\\', '@', '#', '$', '%', '^', '&', '*', '+', '=',
        '<', '>', '|', '~', '`'
    }
    
    if attention is not None:
        attention_weights = attention[0, :actual_length].tolist()
        
        # First pass: collect valid tokens and their weights
        valid_tokens = []
        for token, weight in zip(tokens, attention_weights):
            # Clean up subword tokens for display
            clean_token = token.replace('##', '')
            
            # Skip punctuation and special tokens
            if clean_token in punctuation_tokens:
                continue
            
            # Skip single-character punctuation that might not be in our list
            if len(clean_token) == 1 and not clean_token.isalnum():
                continue
            
            # Skip tokens that are just punctuation
            if clean_token.strip() and not any(c.isalnum() for c in clean_token):
                continue
            
            if clean_token.strip():  # Skip empty tokens
                valid_tokens.append({'word': clean_token, 'weight': weight})
        
        # Renormalize attention weights for valid tokens only
        if valid_tokens:
            total_weight = sum(t['weight'] for t in valid_tokens)
            if total_weight > 0:
                for t in valid_tokens:
                    t['weight'] = t['weight'] / total_weight
        
        attention_data = valid_tokens
    
    return {
        'sentiment': sentiment,
        'confidence': round(confidence * 100, 1),
        'probability': round(probability_positive * 100, 1),
        'attention': attention_data,
        'model': 'Transformer (DistilBERT)'
    }


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/experiment2')
def experiment2():
    """Render the experiment 2 page."""
    return render_template('experiment2.html')


@app.route('/experiment1')
def experiment1():
    """Render the experiment 1 page."""
    return render_template('experiment1.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Analyze the sentiment of the given review."""
    try:
        data = request.get_json()
        review_text = data.get('review', '')
        model_type = data.get('model', 'bilstm')  # Default to bilstm
        
        if not review_text.strip():
            return jsonify({'error': 'Please enter a review'}), 400
        
        # Route to appropriate model
        if model_type == 'transformer':
            if transformer_model is None:
                return jsonify({'error': 'Transformer model not loaded'}), 500
            result = predict_with_transformer(review_text)
        else:
            if bilstm_model is None:
                return jsonify({'error': 'BiLSTM model not loaded'}), 500
            result = predict_with_bilstm(review_text)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/metrics')
def metrics():
    """Return model metrics based on selected model."""
    project_root = Path(__file__).parent.parent
    model_type = request.args.get('model', 'bilstm')
    
    if model_type == 'transformer':
        metrics_path = project_root / "checkpoints" / TransformerConfig.METRICS_FILE
    else:
        metrics_path = project_root / "checkpoints" / "bilstm_metrics.json"
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        return jsonify(metrics_data)
    else:
        return jsonify({'error': 'Metrics not found'}), 404


@app.route('/models')
def get_models():
    """Return available models."""
    models = []
    
    if bilstm_model is not None:
        models.append({
            'id': 'bilstm',
            'name': 'BiLSTM + Attention',
            'description': 'Bidirectional LSTM with Additive Attention'
        })
    
    if transformer_model is not None:
        models.append({
            'id': 'transformer',
            'name': 'Transformer (DistilBERT)',
            'description': 'DistilBERT fine-tuned for sentiment analysis'
        })
    
    return jsonify(models)


if __name__ == '__main__':
    # Load both models
    bilstm_loaded = load_bilstm_model()
    transformer_loaded = load_transformer_model()
    
    if not bilstm_loaded and not transformer_loaded:
        print("WARNING: No models could be loaded!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
