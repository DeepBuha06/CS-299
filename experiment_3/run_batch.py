"""
Batch processing script to run comprehensiveness and sufficiency tests on 100 reviews.
Takes 50 positive and 50 negative reviews from IMDB test dataset.
"""

import torch
import json
import os
from pathlib import Path

from config import ExperimentConfig
from comprehensiveness import ComprehensivenessTester
from sufficiency import SufficiencyTester

# Import from parent directory
import sys
sys.path.insert(0, str(ExperimentConfig.PROJECT_ROOT))

from models.model import AttentionClassifier
from data.preprocessing import Preprocessor


def load_model_and_preprocessor():
    """Load trained BiLSTM model and preprocessor."""
    # Load preprocessor
    preprocessor = Preprocessor.from_vocab_file(
        vocab_path=str(ExperimentConfig.VOCAB_FILE),
        max_length=ExperimentConfig.MAX_SEQ_LENGTH
    )
    
    # Create model architecture
    model = AttentionClassifier(
        vocab_size=ExperimentConfig.VOCAB_SIZE,
        embedding_dim=ExperimentConfig.EMBEDDING_DIM,
        hidden_dim=ExperimentConfig.HIDDEN_DIM,
        attention_dim=ExperimentConfig.ATTENTION_DIM,
        num_classes=ExperimentConfig.NUM_CLASSES,
        num_layers=ExperimentConfig.NUM_LAYERS,
        bidirectional=ExperimentConfig.BIDIRECTIONAL,
        attention_type=ExperimentConfig.ATTENTION_TYPE,
        encoder_dropout=ExperimentConfig.ENCODER_DROPOUT,
        classifier_dropout=ExperimentConfig.CLASSIFIER_DROPOUT,
        padding_idx=ExperimentConfig.PAD_IDX
    )
    
    # Load trained weights
    checkpoint = torch.load(
        str(ExperimentConfig.CHECKPOINT_PATH),
        map_location=ExperimentConfig.DEVICE
    )
    model.load_state_dict(checkpoint)
    model = model.to(ExperimentConfig.DEVICE)
    model.eval()
    
    return model, preprocessor


def get_review_files(directory, count=50):
    """Get first N review files from directory, sorted alphabetically."""
    files = sorted([f for f in os.listdir(directory) if f.endswith('.txt')])
    return files[:count]


def read_review(file_path):
    """Read review text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def run_batch_tests():
    """Run comprehensiveness and sufficiency tests on batch of 100 reviews."""
    
    # Paths
    project_root = ExperimentConfig.PROJECT_ROOT
    pos_dir = project_root / "data" / "raw" / "imdb" / "test" / "pos"
    neg_dir = project_root / "data" / "raw" / "imdb" / "test" / "neg"
    output_file = Path(__file__).parent / "batch_result.json"
    
    # Load model and preprocessor once
    model, preprocessor = load_model_and_preprocessor()
    
    # Initialize testers
    comp_tester = ComprehensivenessTester(model, preprocessor)
    suff_tester = SufficiencyTester(model, preprocessor)
    
    # Get review files (first 50 of each)
    pos_files = get_review_files(pos_dir, 50)
    neg_files = get_review_files(neg_dir, 50)
    
    # Prepare batch
    batch_reviews = []
    for f in pos_files:
        batch_reviews.append({
            'sentiment': 'positive',
            'file': pos_dir / f,
            'filename': f
        })
    for f in neg_files:
        batch_reviews.append({
            'sentiment': 'negative',
            'file': neg_dir / f,
            'filename': f
        })
    
    # Initialize results structure
    batch_results = {
        'total_reviews': len(batch_reviews),
        'reviews_processed': 0,
        'review_details': []
    }
    
    # Process each review
    for idx, review_info in enumerate(batch_reviews):
        try:
            review_text = read_review(review_info['file'])
            
            # Run comprehensiveness test
            comp_results = comp_tester.compute_multiple_k(
                review_text, 
                k_values=ExperimentConfig.TOP_K_VALUES
            )
            
            # Run sufficiency test
            suff_results = suff_tester.compute_multiple_k(
                review_text,
                k_values=ExperimentConfig.TOP_K_VALUES
            )
            
            # Store results
            review_detail = {
                'index': idx,
                'filename': review_info['filename'],
                'sentiment': review_info['sentiment'],
                'comprehensiveness': comp_results,
                'sufficiency': suff_results
            }
            batch_results['review_details'].append(review_detail)
            batch_results['reviews_processed'] += 1
            
        except Exception as e:
            batch_results['review_details'].append({
                'index': idx,
                'filename': review_info['filename'],
                'sentiment': review_info['sentiment'],
                'error': str(e)
            })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(batch_results, f, indent=2, default=str)


if __name__ == '__main__':
    run_batch_tests()
