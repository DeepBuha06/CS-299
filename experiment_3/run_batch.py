
import torch
import json
import os
from pathlib import Path

from config import ExperimentConfig
from comprehensiveness import ComprehensivenessTester
from sufficiency import SufficiencyTester

import sys
sys.path.insert(0, str(ExperimentConfig.PROJECT_ROOT))

from models.model import AttentionClassifier
from data.preprocessing import Preprocessor


def load_model_and_preprocessor():
    preprocessor = Preprocessor.from_vocab_file(
        vocab_path=str(ExperimentConfig.VOCAB_FILE),
        max_length=ExperimentConfig.MAX_SEQ_LENGTH
    )
    
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
    
    checkpoint = torch.load(
        str(ExperimentConfig.CHECKPOINT_PATH),
        map_location=ExperimentConfig.DEVICE
    )
    model.load_state_dict(checkpoint)
    model = model.to(ExperimentConfig.DEVICE)
    model.eval()
    
    return model, preprocessor


def get_review_files(directory, count=50):
    files = sorted([f for f in os.listdir(directory) if f.endswith('.txt')])
    return files[:count]


def read_review(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def run_batch_tests():
    
    project_root = ExperimentConfig.PROJECT_ROOT
    pos_dir = project_root / "data" / "raw" / "imdb" / "test" / "pos"
    neg_dir = project_root / "data" / "raw" / "imdb" / "test" / "neg"
    output_file = Path(__file__).parent / "batch_result.json"
    
    model, preprocessor = load_model_and_preprocessor()
    
    comp_tester = ComprehensivenessTester(model, preprocessor)
    suff_tester = SufficiencyTester(model, preprocessor)
    
    pos_files = get_review_files(pos_dir, 50)
    neg_files = get_review_files(neg_dir, 50)
    
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
    
    batch_results = {
        'total_reviews': len(batch_reviews),
        'reviews_processed': 0,
        'review_details': []
    }
    
    for idx, review_info in enumerate(batch_reviews):
        try:
            review_text = read_review(review_info['file'])
            
            comp_results = comp_tester.compute_multiple_k(
                review_text, 
                k_values=ExperimentConfig.TOP_K_VALUES
            )
            
            suff_results = suff_tester.compute_multiple_k(
                review_text,
                k_values=ExperimentConfig.TOP_K_VALUES
            )
            
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
    
    with open(output_file, 'w') as f:
        json.dump(batch_results, f, indent=2, default=str)


if __name__ == '__main__':
    run_batch_tests()
