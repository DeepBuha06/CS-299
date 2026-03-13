"""
Adversarial Attention Attack
Finds attention distributions that produce the SAME prediction but are DIFFERENT from original attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
import json
from pathlib import Path


class AdversarialAttentionAttack:
    """
    Find adversarial attention weights that give same prediction but different attention.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_difference: float = 0.5,
        max_iterations: int = 100,
        learning_rate: float = 0.01,
        prediction_tolerance: float = 0.05
    ):
        self.model = model
        self.target_difference = target_difference
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.prediction_tolerance = prediction_tolerance
    
    def get_original_attention_and_prediction(
        self,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """Get original attention weights and prediction."""
        self.model.eval()
        with torch.no_grad():
            predictions, attention = self.model(
                token_ids, 
                lengths, 
                return_attention=True
            )
        
        if isinstance(predictions, torch.Tensor):
            prediction = predictions.item()
        else:
            prediction = predictions
        
        attention_weights = attention[0] if attention is not None else None
        return attention_weights, prediction
    
    def find_adversarial_attention_gradient(
        self,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        original_attention: Optional[torch.Tensor] = None,
        original_prediction: Optional[float] = None
    ) -> Tuple[torch.Tensor, float, Dict]:
        """
        Use gradient-based optimization to find adversarial attention.
        
        Optimizes attention weights to maximize attention difference while
        keeping the prediction close to the original.
        """
        if original_attention is None or original_prediction is None:
            original_attention, original_prediction = self.get_original_attention_and_prediction(
                token_ids, lengths
            )
        
        seq_length = original_attention.shape[0]
        
        # Initialize adversarial attention logits from original
        adversarial_logits = original_attention.clone().detach()
        adversarial_logits.requires_grad = True
        
        optimizer = torch.optim.Adam([adversarial_logits], lr=self.learning_rate)
        
        best_attention = original_attention.clone().detach()
        best_difference = 0
        
        # Get hidden states for computing prediction with modified attention
        with torch.no_grad():
            batch_size, seq_len = token_ids.shape
            mask = None
            if lengths is not None:
                range_tensor = torch.arange(seq_len, device=token_ids.device)
                range_tensor = range_tensor.unsqueeze(0).expand(batch_size, -1)
                mask = range_tensor < lengths.unsqueeze(1)
            
            embeddings = self.model.embedding(token_ids)
            hidden_states, _ = self.model.encoder(embeddings, lengths)
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Normalize to valid attention distribution
            if lengths is not None:
                actual_len = lengths[0].item()
                adversarial_logits[actual_len:] = -1e9
                normalized_attention = F.softmax(adversarial_logits, dim=0)
            else:
                normalized_attention = F.softmax(adversarial_logits, dim=0)
            
            # Maximize attention difference
            difference = torch.abs(normalized_attention - original_attention).sum()
            
            # Compute prediction with adversarial attention through the model's classifier
            adv_attention_batch = normalized_attention.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_length)
            context = torch.bmm(adv_attention_batch, hidden_states).squeeze(1)  # (1, hidden_dim)
            adv_prediction = self.model.classifier(context)
            
            # Prediction difference penalty
            prediction_diff = (adv_prediction - original_prediction) ** 2
            
            # Loss: maximize attention difference, minimize prediction difference
            loss = -difference + 100 * prediction_diff.squeeze()
            
            loss.backward()
            optimizer.step()  # Fixed: was missing () call
            
            with torch.no_grad():
                normalized_attention = F.softmax(adversarial_logits, dim=0)
                current_difference = torch.abs(normalized_attention - original_attention).sum().item()
                
                if current_difference > best_difference:
                    best_difference = current_difference
                    best_attention = normalized_attention.clone().detach()
        
        return best_attention, original_prediction, {
            'method': 'gradient',
            'iterations': self.max_iterations,
            'difference': best_difference
        }
    
    def find_adversarial_attention_random(
        self,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        original_attention: Optional[torch.Tensor] = None,
        original_prediction: Optional[float] = None,
        num_samples: int = 1000
    ) -> Tuple[torch.Tensor, float, Dict]:
        """
        Random sampling to find adversarial attention.
        """
        if original_attention is None or original_prediction is None:
            original_attention, original_prediction = self.get_original_attention_and_prediction(
                token_ids, lengths
            )
        
        seq_length = original_attention.shape[0]
        
        best_attention = None
        best_difference = 0
        best_prediction_diff = float('inf')
        
        valid_len = lengths[0].item() if lengths is not None else seq_length
        
        for _ in range(num_samples):
            random_attention = torch.zeros(seq_length, device=original_attention.device)
            if valid_len > 0:
                rand_vals = torch.rand(valid_len, device=original_attention.device)
                random_attention[:valid_len] = rand_vals / rand_vals.sum()
            
            difference = torch.abs(random_attention - original_attention).sum().item()
            
            if difference > best_difference and difference < 2.0:
                best_difference = difference
                best_attention = random_attention.clone()
        
        if best_attention is None:
            best_attention = original_attention.clone()
        
        return best_attention, original_prediction, {
            'method': 'random',
            'samples': num_samples,
            'difference': best_difference
        }
    
    def find_adversarial_attention_permutation(
        self,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        original_attention: Optional[torch.Tensor] = None,
        original_prediction: Optional[float] = None,
        num_permutations: int = 100
    ) -> Tuple[torch.Tensor, float, Dict]:
        """
        Use permutation-based search to find adversarial attention.
        """
        if original_attention is None or original_prediction is None:
            original_attention, original_prediction = self.get_original_attention_and_prediction(
                token_ids, lengths
            )
        
        seq_length = original_attention.shape[0]
        
        best_attention = None
        best_difference = 0
        
        valid_len = lengths[0].item() if lengths is not None else seq_length
        
        original_indices = torch.argsort(original_attention, descending=True)
        
        for _ in range(num_permutations):
            permuted_attention = original_attention.clone()
            
            if valid_len > 1:
                num_swaps = np.random.randint(1, max(2, valid_len // 2 + 1))
                for _ in range(num_swaps):
                    i, j = np.random.choice(valid_len, 2, replace=False)
                    i, j = int(i), int(j)
                    permuted_attention[i], permuted_attention[j] = (
                        permuted_attention[j], permuted_attention[i]
                    )
            
            if valid_len > 0:
                permuted_attention[:valid_len] = permuted_attention[:valid_len] / permuted_attention[:valid_len].sum()
            
            difference = torch.abs(permuted_attention - original_attention).sum().item()
            
            if difference > best_difference:
                best_difference = difference
                best_attention = permuted_attention.clone()
        
        if best_attention is None:
            best_attention = original_attention.clone()
        
        return best_attention, original_prediction, {
            'method': 'permutation',
            'permutations': num_permutations,
            'difference': best_difference
        }
    
    def find_adversarial_attention_entropy(
        self,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        original_attention: Optional[torch.Tensor] = None,
        original_prediction: Optional[float] = None
    ) -> Tuple[torch.Tensor, float, Dict]:
        """
        Find maximum entropy (most uniform) attention that keeps same prediction.
        """
        if original_attention is None or original_prediction is None:
            original_attention, original_prediction = self.get_original_attention_and_prediction(
                token_ids, lengths
            )
        
        seq_length = original_attention.shape[0]
        valid_len = lengths[0].item() if lengths is not None else seq_length
        
        uniform_attention = torch.zeros(seq_length, device=original_attention.device)
        if valid_len > 0:
            uniform_attention[:valid_len] = 1.0 / valid_len
        
        entropy_original = -(original_attention * torch.log(original_attention + 1e-10)).sum()
        entropy_uniform = -(uniform_attention * torch.log(uniform_attention + 1e-10)).sum()
        
        difference = torch.abs(uniform_attention - original_attention).sum().item()
        
        return uniform_attention, original_prediction, {
            'method': 'entropy_maximization',
            'entropy_original': entropy_original.item(),
            'entropy_uniform': entropy_uniform.item(),
            'difference': difference
        }
    
    def find_adversarial_attention_all_methods(
        self,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Run all methods and return best result.
        """
        original_attention, original_prediction = self.get_original_attention_and_prediction(
            token_ids, lengths
        )
        
        results = {}
        
        uniform_attn, pred, info = self.find_adversarial_attention_entropy(
            token_ids, lengths, original_attention, original_prediction
        )
        results['entropy'] = {
            'attention': uniform_attn,
            'prediction': pred,
            'info': info
        }
        
        perm_attn, pred, info = self.find_adversarial_attention_permutation(
            token_ids, lengths, original_attention, original_prediction, num_permutations=200
        )
        results['permutation'] = {
            'attention': perm_attn,
            'prediction': pred,
            'info': info
        }
        
        rand_attn, pred, info = self.find_adversarial_attention_random(
            token_ids, lengths, original_attention, original_prediction, num_samples=500
        )
        results['random'] = {
            'attention': rand_attn,
            'prediction': pred,
            'info': info
        }
        
        best_method = max(results.keys(), 
                         key=lambda k: results[k]['info'].get('difference', 0))
        
        return {
            'original_attention': original_attention,
            'original_prediction': original_prediction,
            'adversarial_attention': results[best_method]['attention'],
            'adversarial_prediction': results[best_method]['prediction'],
            'best_method': best_method,
            'all_results': results,
            'attention_difference': results[best_method]['info'].get('difference', 0),
            'prediction_difference': abs(results[best_method]['prediction'] - original_prediction)
        }


def compute_attention_difference(attention1: torch.Tensor, attention2: torch.Tensor) -> Dict:
    """Compute various metrics for attention difference."""
    diff = torch.abs(attention1 - attention2)
    
    return {
        'l1_difference': diff.sum().item(),
        'l2_difference': torch.sqrt((diff ** 2).sum()).item(),
        'max_difference': diff.max().item(),
        'mean_difference': diff.mean().item(),
        'cosine_similarity': F.cosine_similarity(
            attention1.unsqueeze(0), 
            attention2.unsqueeze(0)
        ).item()
    }


def run_adversarial_experiment(
    model: nn.Module,
    text: str,
    vocab: Dict,
    max_length: int = 256,
    device: str = 'cpu'
) -> Dict:
    """
    Run full adversarial attention experiment on a single text.
    """
    import re
    
    def tokenize(text):
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        words = re.findall(r'\b[a-z]+\b', text)
        return words
    
    tokens = tokenize(text)[:max_length]
    
    ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
    
    if len(ids) < max_length:
        ids.extend([vocab.get('<PAD>', 0)] * (max_length - len(ids)))
    
    length = len(tokens)
    
    token_ids = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([length])
    
    token_ids = token_ids.to(device)
    lengths = lengths.to(device)
    
    attacker = AdversarialAttentionAttack(
        model=model,
        max_iterations=100,
        learning_rate=0.05
    )
    
    result = attacker.find_adversarial_attention_all_methods(token_ids, lengths)
    
    tokens_for_display = tokens[:length]
    
    original_attention_list = result['original_attention'][:length].tolist()
    adversarial_attention_list = result['adversarial_attention'][:length].tolist()
    
    diff_metrics = compute_attention_difference(
        result['original_attention'][:length],
        result['adversarial_attention'][:length]
    )
    
    return {
        'text': text,
        'tokens': tokens_for_display,
        'original_attention': original_attention_list,
        'adversarial_attention': adversarial_attention_list,
        'original_prediction': result['original_prediction'],
        'adversarial_prediction': result['adversarial_prediction'],
        'best_method': result['best_method'],
        'attention_difference': result['attention_difference'],
        'prediction_difference': result['prediction_difference'],
        'difference_metrics': diff_metrics,
        'num_tokens': len(tokens_for_display)
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models.model import AttentionClassifier
    from config import Config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
    
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
    
    checkpoint = torch.load('checkpoints/bilstm_model.pt', map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    test_text = "This movie was absolutely fantastic! Great acting and plot."
    
    result = run_adversarial_experiment(model, test_text, vocab, device=device)
    
    print("=" * 60)
    print("ADVERSARIAL ATTENTION EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Text: {result['text'][:100]}...")
    print(f"\nOriginal Prediction: {result['original_prediction']:.4f}")
    print(f"Adversarial Prediction: {result['adversarial_prediction']:.4f}")
    print(f"\nBest Method: {result['best_method']}")
    print(f"Attention Difference (L1): {result['difference_metrics']['l1_difference']:.4f}")
    print(f"Cosine Similarity: {result['difference_metrics']['cosine_similarity']:.4f}")
    print("\nTop 5 words by attention:")
    for i, (tok, orig, adv) in enumerate(zip(
        result['tokens'][:5],
        result['original_attention'][:5],
        result['adversarial_attention'][:5]
    )):
        print(f"  {tok}: original={orig:.4f}, adversarial={adv:.4f}")
