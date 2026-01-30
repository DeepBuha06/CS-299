"""
Evaluation script for trained model.
Also includes attention visualization.
"""

import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from config import Config
from data.preprocessing import Preprocessor
from models.model import AttentionClassifier
from utils.metrics import calculate_metrics, print_metrics


def load_model(model_path, preprocessor):
    """Load trained model."""
    model = AttentionClassifier(
        vocab_size=preprocessor.vocab_size,
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
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model = model.to(Config.DEVICE)
    model.eval()
    return model


def predict_single(model, preprocessor, text):
    """Predict sentiment for a single text with attention visualization."""
    # Tokenize
    tokens = preprocessor.tokenize(text)
    
    # Process
    token_ids, length = preprocessor.process(text, return_length=True)
    token_ids = token_ids.unsqueeze(0).to(Config.DEVICE)
    lengths = torch.tensor([length]).to(Config.DEVICE)
    
    # Predict
    with torch.no_grad():
        prob, attention = model(token_ids, lengths, return_attention=True)
    
    # Get results
    prediction = "Positive" if prob.item() >= 0.5 else "Negative"
    confidence = prob.item() if prob.item() >= 0.5 else 1 - prob.item()
    
    # Get attention weights for actual tokens
    attention_weights = attention[0, :length].cpu().numpy()
    actual_tokens = tokens[:length]
    
    return {
        "text": text,
        "prediction": prediction,
        "confidence": confidence,
        "probability": prob.item(),
        "tokens": actual_tokens,
        "attention": attention_weights
    }


def visualize_attention(result, save_path=None):
    """Visualize attention weights."""
    tokens = result["tokens"][:20]  # Limit for visibility
    attention = result["attention"][:20]
    
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Create bar chart
    x = np.arange(len(tokens))
    bars = ax.bar(x, attention, color='steelblue', alpha=0.7)
    
    # Highlight top attention
    max_idx = np.argmax(attention)
    bars[max_idx].set_color('crimson')
    
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    # Load preprocessor
    print("Loading vocabulary...")
    preprocessor = Preprocessor.from_vocab_file(
        Config.VOCAB_FILE,
        max_length=Config.MAX_SEQ_LENGTH
    )
    
    # Load model
    model_path = Config.MODEL_DIR / "best_model.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    print("Loading model...")
    model = load_model(model_path, preprocessor)
    
    # Test examples
    test_texts = [
        "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout.",
        "Terrible waste of time. The plot made no sense and the acting was wooden.",
        "It was okay, nothing special but not bad either.",
    ]
    
    print("\n" + "=" * 60)
    print("PREDICTIONS")
    print("=" * 60)
    
    for i, text in enumerate(test_texts):
        result = predict_single(model, preprocessor, text)
        print(f"\nText: {text[:80]}...")
        print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
        print(f"Top attended words: ", end="")
        
        # Show top 5 attended words
        indices = np.argsort(result["attention"])[-5:][::-1]
        for idx in indices:
            if idx < len(result["tokens"]):
                print(f"{result['tokens'][idx]} ({result['attention'][idx]:.3f})", end=" ")
        print()
        
        # Save attention visualization
        save_path = f"attention_viz_{i+1}.png"
        visualize_attention(result, save_path=save_path)


if __name__ == "__main__":
    main()
