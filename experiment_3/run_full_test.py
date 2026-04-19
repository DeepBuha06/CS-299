
import torch
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import tqdm

from config import ExperimentConfig
from comprehensiveness import ComprehensivenessTester
from sufficiency import SufficiencyTester

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
    
    try:
        checkpoint = torch.load(
            str(ExperimentConfig.CHECKPOINT_PATH),
            map_location=ExperimentConfig.DEVICE,
            weights_only=False
        )
        model.load_state_dict(checkpoint)
        print(f"✓ Successfully loaded checkpoint from {ExperimentConfig.CHECKPOINT_PATH}")
    except Exception as e:
        print(f"⚠ Warning: Could not load checkpoint ({str(e)})")
        print(f"  Using randomly initialized model weights instead")
    
    model = model.to(ExperimentConfig.DEVICE)
    model.eval()
    
    return model, preprocessor


def get_all_review_files(pos_dir, neg_dir):
    
    pos_files = sorted([f for f in os.listdir(pos_dir) if f.endswith('.txt')])
    pos_files = [{'path': pos_dir / f, 'filename': f, 'sentiment': 'positive'} 
                 for f in pos_files]
    
    neg_files = sorted([f for f in os.listdir(neg_dir) if f.endswith('.txt')])
    neg_files = [{'path': neg_dir / f, 'filename': f, 'sentiment': 'negative'} 
                 for f in neg_files]
    
    return pos_files + neg_files


def read_review(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {str(e)}")


def run_full_test(output_file=None):
    
    if output_file is None:
        output_file = Path(__file__).parent / "results" / "full_test_results.json"
    else:
        output_file = Path(output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    project_root = ExperimentConfig.PROJECT_ROOT
    pos_dir = project_root / "data" / "raw" / "imdb" / "test" / "pos"
    neg_dir = project_root / "data" / "raw" / "imdb" / "test" / "neg"
    
    print(f"Starting full test on all reviews in IMDB test dataset...")
    print(f"Positive reviews directory: {pos_dir}")
    print(f"Negative reviews directory: {neg_dir}")
    
    all_reviews = get_all_review_files(pos_dir, neg_dir)
    print(f"Total reviews found: {len(all_reviews)}")
    
    print("Loading model and preprocessor...")
    model, preprocessor = load_model_and_preprocessor()
    
    comp_tester = ComprehensivenessTester(model, preprocessor)
    suff_tester = SufficiencyTester(model, preprocessor)
    
    full_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_reviews': len(all_reviews),
            'top_k_values': ExperimentConfig.TOP_K_VALUES,
            'device': str(ExperimentConfig.DEVICE),
            'model_checkpoint': str(ExperimentConfig.CHECKPOINT_PATH)
        },
        'reviews_processed': 0,
        'reviews_failed': 0,
        'review_details': []
    }
    
    print(f"\nProcessing {len(all_reviews)} reviews...")
    print("=" * 80)
    
    for idx, review_info in enumerate(tqdm(all_reviews, desc="Processing reviews")):
        try:
            review_text = read_review(review_info['path'])
            
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
            full_results['review_details'].append(review_detail)
            full_results['reviews_processed'] += 1
            
        except Exception as e:
            full_results['review_details'].append({
                'index': idx,
                'filename': review_info['filename'],
                'sentiment': review_info['sentiment'],
                'error': str(e),
                'status': 'failed'
            })
            full_results['reviews_failed'] += 1
            print(f"\nError processing {review_info['filename']}: {str(e)}")
    
    print("=" * 80)
    print(f"\nTest completed!")
    print(f"Reviews processed: {full_results['reviews_processed']}")
    print(f"Reviews failed: {full_results['reviews_failed']}")
    
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    print(f"Results saved successfully!")
    print(f"Total file size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    
    return full_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run comprehensiveness and sufficiency tests on all IMDB test reviews'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (default: results/full_test_results.json)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use: auto, cpu, or cuda (default: auto)'
    )
    
    args = parser.parse_args()
    
    if args.device != 'auto':
        if args.device.lower() == 'cpu':
            ExperimentConfig.DEVICE = torch.device('cpu')
        elif args.device.lower() == 'cuda':
            ExperimentConfig.DEVICE = torch.device('cuda')
    
    print(f"Using device: {ExperimentConfig.DEVICE}")
    
    run_full_test(output_file=args.output)
