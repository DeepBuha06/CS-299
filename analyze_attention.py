"""
Analyze attention weights across test data to find words 
that contribute most to positive vs negative predictions.
"""

import json
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from config import Config
from data.preprocessing import Preprocessor
from data.dataset import IMDBDataset
from models.model import AttentionClassifier


def load_model(model_path, vocab_size):
    """Load trained model."""
    model = AttentionClassifier(
        vocab_size=vocab_size,
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


def analyze_attention(model, preprocessor, dataset, num_samples=1000):
    """
    Analyze attention weights to find top contributing words.
    
    Returns:
        positive_words: dict of word -> average attention when predicting positive
        negative_words: dict of word -> average attention when predicting negative
    """
    # Track attention weights for each word
    positive_attention = defaultdict(list)  # word -> list of attention weights
    negative_attention = defaultdict(list)
    
    print(f"Analyzing {num_samples} samples...")
    
    for i in tqdm(range(min(num_samples, len(dataset)))):
        token_ids, label, length = dataset[i]
        token_ids = token_ids.unsqueeze(0).to(Config.DEVICE)
        lengths = torch.tensor([length]).to(Config.DEVICE)
        
        # Get prediction and attention
        with torch.no_grad():
            prob, attention = model(token_ids, lengths, return_attention=True)
        
        prediction = prob.item() >= 0.5
        attention_weights = attention[0, :length].cpu().numpy()
        
        # Get tokens for this review
        review_text = dataset.reviews[i]
        tokens = preprocessor.tokenize(review_text)[:length]
        
        # Aggregate attention by word
        for token, weight in zip(tokens, attention_weights):
            if prediction:  # Predicted positive
                positive_attention[token].append(weight)
            else:  # Predicted negative
                negative_attention[token].append(weight)
    
    # Calculate average attention for each word
    positive_avg = {word: np.mean(weights) for word, weights in positive_attention.items() if len(weights) >= 5}
    negative_avg = {word: np.mean(weights) for word, weights in negative_attention.items() if len(weights) >= 5}
    
    return positive_avg, negative_avg


def main():
    # Load preprocessor
    print("Loading vocabulary...")
    preprocessor = Preprocessor.from_vocab_file(
        Config.VOCAB_FILE,
        max_length=Config.MAX_SEQ_LENGTH
    )
    
    # Load model
    model_path = Config.MODEL_DIR / "best_model.pt"
    print("Loading model...")
    model = load_model(model_path, preprocessor.vocab_size)
    
    # Load test dataset
    print("Loading test data...")
    test_dataset = IMDBDataset(Config.DATA_DIR, preprocessor, split="test")
    
    # Analyze attention
    positive_words, negative_words = analyze_attention(
        model, preprocessor, test_dataset, num_samples=2000
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("TOP WORDS FOR POSITIVE PREDICTIONS")
    print("=" * 60)
    top_positive = sorted(positive_words.items(), key=lambda x: x[1], reverse=True)[:30]
    for word, score in top_positive:
        print(f"  {word:20s} {score:.4f}")
    
    print("\n" + "=" * 60)
    print("TOP WORDS FOR NEGATIVE PREDICTIONS")
    print("=" * 60)
    top_negative = sorted(negative_words.items(), key=lambda x: x[1], reverse=True)[:30]
    for word, score in top_negative:
        print(f"  {word:20s} {score:.4f}")
    
    # Save results (convert numpy floats to Python floats)
    results = {
        "positive_words": {k: float(v) for k, v in top_positive},
        "negative_words": {k: float(v) for k, v in top_negative}
    }
    with open("attention_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to attention_analysis.json")


if __name__ == "__main__":
    main()
