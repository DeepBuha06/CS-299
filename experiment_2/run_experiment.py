"""
Run full adversarial attention experiment pipeline.
"""

import torch
import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import AttentionClassifier
from config import Config

try:
    from adversarial_attack import run_adversarial_experiment
    from visualization import AdversarialVisualizer
    from comparison import AttentionComparator
except ImportError:
    from experiment_1.adversarial_attack import run_adversarial_experiment
    from experiment_1.visualization import AdversarialVisualizer
    from experiment_1.comparison import AttentionComparator


class ExperimentRunner:
    """Run the complete adversarial attention experiment."""
    
    def __init__(self, model_path: str, vocab_path: str, device: str = 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vocab = self._load_vocab(vocab_path)
        self.model = self._load_model(model_path)
        self.visualizer = AdversarialVisualizer()
    
    def _load_vocab(self, vocab_path: str) -> Dict:
        """Load vocabulary from JSON file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_model(self, model_path: str) -> AttentionClassifier:
        """Load trained model from checkpoint."""
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
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        return model
    
    def run_single_text(self, text: str) -> Dict:
        """Run experiment on a single text."""
        result = run_adversarial_experiment(
            model=self.model,
            text=text,
            vocab=self.vocab,
            max_length=Config.MAX_SEQ_LENGTH,
            device=str(self.device)
        )
        
        visualizations = self.visualizer.generate_full_visualization(
            tokens=result['tokens'],
            original_attention=result['original_attention'],
            adversarial_attention=result['adversarial_attention'],
            metrics=result['difference_metrics']
        )
        
        report, comparison_details = AttentionComparator.generate_comparison_report(
            tokens=result['tokens'],
            original_attention=result['original_attention'],
            adversarial_attention=result['adversarial_attention'],
            original_prediction=result['original_prediction'],
            adversarial_prediction=result['adversarial_prediction']
        )
        
        result['visualizations'] = visualizations
        result['comparison_report'] = report
        result['comparison_details'] = comparison_details
        
        return result
    
    def run_batch(self, texts: List[str]) -> Dict:
        """Run experiment on multiple texts."""
        results = []
        
        for i, text in enumerate(texts):
            print(f"Processing text {i+1}/{len(texts)}...")
            result = self.run_single_text(text)
            results.append(result)
        
        batch_stats = self._compute_batch_statistics(results)
        
        return {
            'individual_results': results,
            'batch_statistics': batch_stats
        }
    
    def _compute_batch_statistics(self, results: List[Dict]) -> Dict:
        """Compute aggregate statistics across all results."""
        from comparison import batch_compare_attentions
        
        batch_results = []
        for r in results:
            batch_results.append({
                'original_attention': r['original_attention'],
                'adversarial_attention': r['adversarial_attention'],
                'original_prediction': r['original_prediction'],
                'adversarial_prediction': r['adversarial_prediction']
            })
        
        stats = batch_compare_attentions(batch_results)
        
        sentiment_labels = {
            'positive': 0,
            'negative': 0,
            'same': 0,
            'changed': 0
        }
        
        for r in results:
            orig_sent = 'positive' if r['original_prediction'] >= 0.5 else 'negative'
            adv_sent = 'positive' if r['adversarial_prediction'] >= 0.5 else 'negative'
            
            sentiment_labels[orig_sent] += 1
            if orig_sent == adv_sent:
                sentiment_labels['same'] += 1
            else:
                sentiment_labels['changed'] += 1
        
        stats['sentiment_distribution'] = sentiment_labels
        
        return stats
    
    def get_sample_texts(self) -> Dict[str, str]:
        """Get sample texts for testing."""
        return {
            'positive_strong': "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. Highly recommend!",
            'positive_moderate': "I enjoyed watching this film. It had some good moments and the characters were interesting.",
            'negative_strong': "Terrible movie! Complete waste of time. Bad acting, boring storyline, and poor production quality.",
            'negative_moderate': "The film was not very good. It had some issues with pacing and the script could have been better.",
            'neutral': "The movie was okay. Nothing special but not terrible either. Average entertainment."
        }


def main():
    """Main entry point for running the experiment."""
    
    project_root = Path(__file__).parent.parent
    
    model_path = project_root / "checkpoints" / "bilstm_model.pt"
    vocab_path = project_root / "vocab.json"
    
    print("=" * 70)
    print("ADVERSARIAL ATTENTION EXPERIMENT")
    print("=" * 70)
    
    runner = ExperimentRunner(
        model_path=str(model_path),
        vocab_path=str(vocab_path),
        device='cpu'
    )
    
    sample_texts = runner.get_sample_texts()
    
    print("\n" + "=" * 70)
    print("Running experiments on sample texts...")
    print("=" * 70)
    
    results = runner.run_batch(list(sample_texts.values()))
    
    print("\n" + "=" * 70)
    print("BATCH STATISTICS")
    print("=" * 70)
    
    stats = results['batch_statistics']
    print(f"Number of samples: {stats['num_samples']}")
    print(f"Average L1 difference: {stats['average_l1_difference']:.4f}")
    print(f"Average correlation: {stats['average_correlation']:.4f}")
    print(f"Same prediction rate: {stats['same_prediction_rate']*100:.1f}%")
    print(f"Sentiment changes: {stats['sentiment_distribution']['changed']}")
    
    print("\n" + "=" * 70)
    print("INDIVIDUAL RESULTS")
    print("=" * 70)
    
    for i, (name, text) in enumerate(sample_texts.items()):
        result = results['individual_results'][i]
        print(f"\n--- {name.upper()} ---")
        print(f"Text: {text[:60]}...")
        print(f"Original prediction: {result['original_prediction']:.4f}")
        print(f"Adversarial prediction: {result['adversarial_prediction']:.4f}")
        print(f"Attention difference: {result['difference_metrics']['l1_difference']:.4f}")
        print(f"Correlation: {result['difference_metrics']['cosine_similarity']:.4f}")
    
    output_path = project_root / "experiment_1" / "experiment_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    main()
