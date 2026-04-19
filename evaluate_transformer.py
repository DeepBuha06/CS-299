import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from config_transformer import TransformerConfig
from models_transformer.model import TransformerClassifier
from models_transformer.dataset import get_tokenizer


def load_model(model_path: str) -> TransformerClassifier:
    model = TransformerClassifier(
        model_name=TransformerConfig.MODEL_NAME,
        num_labels=TransformerConfig.NUM_LABELS
    )
    
    checkpoint = torch.load(model_path, map_location=TransformerConfig.DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(TransformerConfig.DEVICE)
    model.eval()
    return model


def predict_single(
    model: TransformerClassifier,
    tokenizer,
    text: str,
    return_attention: bool = True
) -> dict:
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=TransformerConfig.MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(TransformerConfig.DEVICE)
    attention_mask = encoding["attention_mask"].to(TransformerConfig.DEVICE)
    
    with torch.no_grad():
        logits, attention = model(input_ids, attention_mask, return_attention=return_attention)
        probabilities = torch.softmax(logits, dim=-1)
    
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class].item()
    
    prediction = "Positive" if predicted_class == 1 else "Negative"
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    actual_length = attention_mask[0].sum().item()
    tokens = tokens[:actual_length]
    
    result = {
        "text": text,
        "prediction": prediction,
        "confidence": confidence,
        "probability_positive": probabilities[0, 1].item(),
        "probability_negative": probabilities[0, 0].item(),
        "tokens": tokens
    }
    
    if return_attention and attention is not None:
        attention_weights = attention[0, :actual_length].cpu().numpy()
        result["attention"] = attention_weights
    
    return result


def visualize_attention(result: dict, save_path: str = None, top_n: int = 20):
    if "attention" not in result:
        print("No attention weights available")
        return
    
    tokens = result["tokens"]
    attention = result["attention"]
    
    if len(tokens) > top_n:
        top_indices = np.argsort(attention)[-top_n:]
        tokens = [tokens[i] for i in top_indices]
        attention = attention[top_indices]
        sorted_indices = np.argsort(top_indices)
        tokens = [tokens[i] for i in sorted_indices]
        attention = attention[sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    x = np.arange(len(tokens))
    bars = ax.bar(x, attention, color='steelblue', alpha=0.7)
    
    max_idx = np.argmax(attention)
    bars[max_idx].set_color('crimson')
    
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Attention Weight')
    ax.set_title(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_top_attention_tokens(result: dict, top_n: int = 10):
    if "attention" not in result:
        print("No attention weights available")
        return
    
    tokens = result["tokens"]
    attention = result["attention"]
    
    top_indices = np.argsort(attention)[-top_n:][::-1]
    
    print(f"\nTop {top_n} attended tokens:")
    for i, idx in enumerate(top_indices):
        if idx < len(tokens):
            print(f"  {i+1}. '{tokens[idx]}': {attention[idx]:.4f}")


def main():
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(TransformerConfig.MODEL_NAME)
    
    model_path = TransformerConfig.MODEL_DIR / TransformerConfig.MODEL_CHECKPOINT
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first using train_transformer.py")
        return
    
    print("Loading model...")
    model = load_model(str(model_path))
    print("Model loaded successfully!")
    
    test_texts = [
        "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout.",
        "Terrible waste of time. The plot made no sense and the acting was wooden.",
        "It was okay, nothing special but not bad either. Some good moments but overall forgettable.",
        "A masterpiece of cinema! Beautiful visuals, compelling story, and unforgettable performances.",
        "I couldn't even finish watching it. Boring, predictable, and poorly executed."
    ]
    
    print("\n" + "=" * 60)
    print("=" * 60)
    
    for i, text in enumerate(test_texts):
        result = predict_single(model, tokenizer, text)
        
        print(f"\n--- Example {i+1} ---")
        print(f"Text: {text[:80]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"P(Positive): {result['probability_positive']:.2%}")
        print(f"P(Negative): {result['probability_negative']:.2%}")
        
        print_top_attention_tokens(result, top_n=5)
        
        save_path = f"transformer_attention_{i+1}.png"
        visualize_attention(result, save_path=save_path)
    
    metrics_path = TransformerConfig.MODEL_DIR / TransformerConfig.METRICS_FILE
    if metrics_path.exists():
        print("\n" + "=" * 60)
        print("=" * 60)
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
