"""
Build vocabulary from IMDB dataset.
Creates vocab.json with the most frequent words from train reviews.
"""

import os
import re
import json
from collections import Counter
from pathlib import Path


def tokenize(text):
    """Simple tokenizer: lowercase and extract words."""
    text = text.lower()
    # Remove HTML tags like <br />
    text = re.sub(r'<[^>]+>', ' ', text)
    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b[a-z]+\b', text)
    return words


def load_reviews(data_dir):
    """Load all reviews from neg and pos folders."""
    reviews = []
    data_path = Path(data_dir)
    
    for folder in ['neg', 'pos']:
        folder_path = data_path / folder
        if not folder_path.exists():
            print(f"Warning: {folder_path} does not exist")
            continue
            
        files = list(folder_path.glob('*.txt'))
        print(f"Loading {len(files)} files from {folder}...")
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                reviews.append(f.read())
    
    return reviews


def build_vocab(reviews, vocab_size=1000):
    """Build vocabulary from reviews."""
    word_counts = Counter()
    
    print("Tokenizing reviews...")
    for i, review in enumerate(reviews):
        words = tokenize(review)
        word_counts.update(words)
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(reviews)} reviews...")
    
    print(f"Total unique words: {len(word_counts)}")
    
    # Get most common words
    most_common = word_counts.most_common(vocab_size)
    
    # Create vocab dict with special tokens
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
    }
    
    for idx, (word, count) in enumerate(most_common, start=2):
        vocab[word] = idx
    
    return vocab, word_counts


def main():
    # Configuration
    DATA_DIR = 'data/raw/imdb/train'
    VOCAB_SIZE = 15000
    OUTPUT_FILE = 'vocab.json'
    
    print(f"Building vocabulary from {DATA_DIR}")
    print(f"Target vocab size: {VOCAB_SIZE}")
    print("-" * 50)
    
    # Load reviews
    reviews = load_reviews(DATA_DIR)
    print(f"Loaded {len(reviews)} reviews total")
    print("-" * 50)
    
    # Build vocabulary
    vocab, word_counts = build_vocab(reviews, vocab_size=VOCAB_SIZE)
    
    print("-" * 50)
    print(f"Final vocabulary size: {len(vocab)} (including special tokens)")
    
    # Save vocabulary
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Saved vocabulary to {OUTPUT_FILE}")
    
    # Print some stats
    print("\nTop 20 words:")
    for word, count in word_counts.most_common(20):
        print(f"  {word}: {count}")


if __name__ == '__main__':
    main()
