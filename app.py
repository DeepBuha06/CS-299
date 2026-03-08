"""
Gradio App for Hugging Face Spaces
Sentiment Analysis with Attention Visualization
"""

import gradio as gr
import torch
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from models.model import AttentionClassifier
from config import Config
from config_transformer import TransformerConfig
from models_transformer.model import TransformerClassifier
from models_transformer.dataset import get_tokenizer


# Global variables
bilstm_model = None
transformer_model = None
vocab = None
transformer_tokenizer = None


def tokenize(text):
    """Simple tokenizer for BiLSTM."""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    words = re.findall(r'\b[a-z]+\b', text)
    return words


def text_to_tensor(text, vocab, max_length=256):
    """Convert text to tensor for BiLSTM."""
    tokens = tokenize(text)
    ids = [vocab.get(token, Config.UNK_IDX) for token in tokens[:max_length]]
    length = len(ids)
    if length < max_length:
        ids.extend([Config.PAD_IDX] * (max_length - length))
    return torch.tensor([ids]), torch.tensor([length])


def load_bilstm_model():
    """Load BiLSTM model."""
    global bilstm_model, vocab
    
    try:
        # Load vocabulary
        vocab_path = Path("vocab.json")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        # Create model
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
        
        # Load weights
        model_path = Path("checkpoints/bilstm_model.pt")
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            bilstm_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            bilstm_model.load_state_dict(checkpoint)
        
        bilstm_model.eval()
        return True
    except Exception as e:
        print(f"Error loading BiLSTM: {e}")
        return False


def load_transformer_model():
    """Load Transformer model."""
    global transformer_model, transformer_tokenizer
    
    try:
        transformer_tokenizer = get_tokenizer(TransformerConfig.MODEL_NAME)
        
        transformer_model = TransformerClassifier(
            model_name=TransformerConfig.MODEL_NAME,
            num_labels=TransformerConfig.NUM_LABELS
        )
        
        model_path = Path("checkpoints") / TransformerConfig.MODEL_CHECKPOINT
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            transformer_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            transformer_model.load_state_dict(checkpoint)
        
        transformer_model.eval()
        return True
    except Exception as e:
        print(f"Error loading Transformer: {e}")
        return False


def create_attention_plot(tokens, attention_weights, title):
    """Create attention visualization plot."""
    # Limit to top 20 tokens for visibility
    if len(tokens) > 20:
        top_indices = np.argsort(attention_weights)[-20:]
        tokens = [tokens[i] for i in top_indices]
        attention_weights = [attention_weights[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['crimson' if w == max(attention_weights) else 'steelblue' for w in attention_weights]
    bars = ax.barh(range(len(tokens)), attention_weights, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.set_xlabel('Attention Weight', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def predict_bilstm(text):
    """Predict with BiLSTM model."""
    if bilstm_model is None or vocab is None:
        return "BiLSTM model not loaded", None
    
    token_ids, lengths = text_to_tensor(text, vocab, Config.MAX_SEQ_LENGTH)
    
    with torch.no_grad():
        predictions, attention_weights = bilstm_model(token_ids, lengths, return_attention=True)
        probability = predictions.item()
    
    sentiment = 'Positive 😊' if probability >= 0.5 else 'Negative 😞'
    confidence = probability if probability >= 0.5 else (1 - probability)
    
    result_text = f"""
**Sentiment**: {sentiment}
**Confidence**: {confidence*100:.1f}%
**Probability (Positive)**: {probability*100:.1f}%
**Model**: BiLSTM + Attention
"""
    
    # Create attention plot
    tokens = tokenize(text)[:Config.MAX_SEQ_LENGTH]
    if attention_weights is not None and len(tokens) > 0:
        weights = attention_weights[0][:len(tokens)].numpy()
        plot = create_attention_plot(tokens, weights, f"BiLSTM Attention Weights - {sentiment}")
    else:
        plot = None
    
    return result_text, plot


def predict_transformer(text):
    """Predict with Transformer model."""
    if transformer_model is None or transformer_tokenizer is None:
        return "Transformer model not loaded", None
    
    encoding = transformer_tokenizer(
        text,
        truncation=True,
        max_length=TransformerConfig.MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    
    with torch.no_grad():
        logits, attention = transformer_model(
            encoding["input_ids"], 
            encoding["attention_mask"], 
            return_attention=True
        )
        probabilities = torch.softmax(logits, dim=-1)
    
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class].item()
    probability_positive = probabilities[0, 1].item()
    
    sentiment = 'Positive 😊' if predicted_class == 1 else 'Negative 😞'
    
    result_text = f"""
**Sentiment**: {sentiment}
**Confidence**: {confidence*100:.1f}%
**Probability (Positive)**: {probability_positive*100:.1f}%
**Model**: Transformer (DistilBERT)
"""
    
    # Create attention plot
    tokens = transformer_tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    actual_length = encoding["attention_mask"][0].sum().item()
    tokens = tokens[:actual_length]
    
    if attention is not None:
        attention_weights = attention[0, :actual_length].numpy()
        
        # Filter out special tokens
        filtered_tokens = []
        filtered_weights = []
        for token, weight in zip(tokens, attention_weights):
            clean_token = token.replace('##', '')
            if clean_token not in ['[CLS]', '[SEP]', '[PAD]'] and clean_token.strip():
                filtered_tokens.append(clean_token)
                filtered_weights.append(weight)
        
        if filtered_tokens:
            # Renormalize
            total = sum(filtered_weights)
            filtered_weights = [w/total for w in filtered_weights]
            plot = create_attention_plot(filtered_tokens, filtered_weights, f"Transformer Attention Weights - {sentiment}")
        else:
            plot = None
    else:
        plot = None
    
    return result_text, plot


def analyze_sentiment(text, model_choice):
    """Main prediction function."""
    if not text.strip():
        return "⚠️ Please enter a movie review to analyze.", None
    
    if model_choice == "BiLSTM + Attention":
        return predict_bilstm(text)
    else:
        return predict_transformer(text)


# Load models at startup
print("Loading models...")
bilstm_loaded = load_bilstm_model()
transformer_loaded = load_transformer_model()

if bilstm_loaded:
    print("✅ BiLSTM model loaded")
if transformer_loaded:
    print("✅ Transformer model loaded")

# Create Gradio interface
with gr.Blocks(title="Sentiment Analysis - Attention Visualization", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎬 Sentiment Analysis with Attention Visualization
        ### Research Project: "Attention is not Explanation"
        
        Compare BiLSTM and Transformer models on IMDB movie reviews. 
        See which words the models *attend to* - but remember, **attention ≠ explanation**!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter Movie Review",
                placeholder="Type or paste your movie review here...",
                lines=6
            )
            
            model_choice = gr.Radio(
                choices=["BiLSTM + Attention", "Transformer (DistilBERT)"],
                value="BiLSTM + Attention",
                label="Select Model"
            )
            
            analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            result_output = gr.Markdown(label="Results")
    
    with gr.Row():
        plot_output = gr.Plot(label="Attention Weights Visualization")
    
    # Examples
    gr.Examples(
        examples=[
            ["This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.", "BiLSTM + Attention"],
            ["Terrible film. Waste of time and money. Poor acting and boring storyline.", "BiLSTM + Attention"],
            ["A masterpiece of cinema. Beautiful cinematography and emotional performances.", "Transformer (DistilBERT)"],
            ["Not my cup of tea. Found it slow and predictable with unlikeable characters.", "Transformer (DistilBERT)"],
        ],
        inputs=[text_input, model_choice],
        label="Example Reviews"
    )
    
    # Model info
    gr.Markdown(
        """
        ---
        ### 📊 Model Performance (IMDB Dataset)
        
        | Model | Accuracy | F1 Score | Notes |
        |-------|----------|----------|-------|
        | BiLSTM + Attention | 86.2% | 85.4% | Custom implementation |
        | Transformer (DistilBERT) | 90.8% | 90.9% | Fine-tuned pre-trained model |
        
        **Dataset**: 50,000 IMDB movie reviews (binary sentiment classification)
        
        ---
        
        ⚠️ **Research Note**: This project tests whether attention weights provide faithful explanations. 
        High attention on a word doesn't necessarily mean it's causally important for the prediction!
        
        📚 **Paper**: [Attention is not Explanation (Jain & Wallace, 2019)](https://arxiv.org/abs/1902.10186)
        
        🔗 **GitHub**: [Ramji-Purwar/CS-299](https://github.com/Ramji-Purwar/CS-299)
        """
    )
    
    # Connect button
    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=[text_input, model_choice],
        outputs=[result_output, plot_output]
    )

# Launch
if __name__ == "__main__":
    demo.launch()
