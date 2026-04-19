import torch
import torch.nn.functional as F
import json
import sys
import time
import re
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from models.model import AttentionClassifier
from data.preprocessing import Preprocessor
from data.dataset import IMDBDataset, collate_fn

from tqdm import tqdm


def get_hidden_states_and_attention(
    model: AttentionClassifier,
    token_ids: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    
    model.eval()
    with torch.no_grad():
        token_ids = token_ids.to(device)
        lengths = lengths.to(device)

        predictions, attention_weights = model(token_ids, lengths, return_attention=True)

        # hidden states for adversarial re-scoring
        embeddings = model.embedding(token_ids)
        hidden_states, _ = model.encoder(embeddings, lengths)

    prediction_value = predictions[0].item()
    original_attention = attention_weights[0]  # (seq_length,)

    return hidden_states, original_attention, prediction_value


def adversarial_entropy(
    original_attention: torch.Tensor,
    seq_length: int,
    valid_len: int,
    device: torch.device
) -> Tuple[torch.Tensor, Dict]:

    uniform = torch.zeros(seq_length, device=device)
    if valid_len > 0:
        uniform[:valid_len] = 1.0 / valid_len

    diff = torch.abs(uniform - original_attention).sum().item()
    return uniform, {'method': 'entropy', 'difference': diff}


def adversarial_permutation(
    original_attention: torch.Tensor,
    seq_length: int,
    valid_len: int,
    device: torch.device,
    num_permutations: int = 100
) -> Tuple[torch.Tensor, Dict]:

    best_attention = None
    best_diff = 0.0

    orig_cpu = original_attention[:valid_len].cpu().numpy()

    for _ in range(num_permutations):
        perm = orig_cpu.copy()
        if valid_len > 1:
            num_swaps = np.random.randint(1, max(2, valid_len // 2 + 1))
            for _ in range(num_swaps):
                i, j = np.random.choice(valid_len, 2, replace=False)
                perm[i], perm[j] = perm[j], perm[i]

        perm_sum = perm.sum()
        if perm_sum > 0:
            perm = perm / perm_sum

        perm_tensor = torch.zeros(seq_length, device=device)
        perm_tensor[:valid_len] = torch.tensor(perm, device=device, dtype=torch.float32)

        diff = torch.abs(perm_tensor - original_attention).sum().item()
        if diff > best_diff:
            best_diff = diff
            best_attention = perm_tensor.clone()

    if best_attention is None:
        best_attention = original_attention.clone()

    return best_attention, {'method': 'permutation', 'difference': best_diff}


def adversarial_random(
    original_attention: torch.Tensor,
    seq_length: int,
    valid_len: int,
    device: torch.device,
    num_samples: int = 500
) -> Tuple[torch.Tensor, Dict]:
 
    best_attention = None
    best_diff = 0.0

    for _ in range(num_samples):
        rand_attn = torch.zeros(seq_length, device=device)
        if valid_len > 0:
            rand_vals = torch.rand(valid_len, device=device)
            rand_attn[:valid_len] = rand_vals / rand_vals.sum()

        diff = torch.abs(rand_attn - original_attention).sum().item()
        if diff > best_diff and diff < 2.0:
            best_diff = diff
            best_attention = rand_attn.clone()

    if best_attention is None:
        best_attention = original_attention.clone()

    return best_attention, {'method': 'random', 'difference': best_diff}


def compute_adversarial_prediction(
    model: AttentionClassifier,
    hidden_states: torch.Tensor,
    adv_attention: torch.Tensor,
    device: torch.device
) -> float:
 
    model.eval()
    with torch.no_grad():
        # adv_attention: (seq_length,) -> (1, 1, seq_length) for bmm
        adv_attn_batch = adv_attention.unsqueeze(0).unsqueeze(1)
        # hidden_states: (1, seq_length, hidden_dim)
        context = torch.bmm(adv_attn_batch, hidden_states).squeeze(1)  # (1, hidden_dim)
        pred = model.classifier(context)
    return pred.item()


def run_attack_single_sample(
    model: AttentionClassifier,
    token_ids: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device
) -> Dict:
 
    hidden_states, original_attention, original_prediction = get_hidden_states_and_attention(
        model, token_ids, lengths, device
    )

    seq_length = original_attention.shape[0]
    valid_len = int(lengths[0].item())

    entropy_attn, entropy_info = adversarial_entropy(original_attention, seq_length, valid_len, device)
    perm_attn, perm_info = adversarial_permutation(original_attention, seq_length, valid_len, device, num_permutations=100)
    rand_attn, rand_info = adversarial_random(original_attention, seq_length, valid_len, device, num_samples=500)

    methods = {
        'entropy': (entropy_attn, entropy_info),
        'permutation': (perm_attn, perm_info),
        'random': (rand_attn, rand_info)
    }
    best_method = max(methods.keys(), key=lambda k: methods[k][1]['difference'])
    best_adv_attention = methods[best_method][0]
    best_diff = methods[best_method][1]['difference']

    adv_prediction = compute_adversarial_prediction(model, hidden_states, best_adv_attention, device)

    orig_trimmed = original_attention[:valid_len]
    adv_trimmed = best_adv_attention[:valid_len]

    diff_tensor = torch.abs(orig_trimmed - adv_trimmed)
    l1_diff = diff_tensor.sum().item()
    l2_diff = torch.sqrt((diff_tensor ** 2).sum()).item()
    max_diff = diff_tensor.max().item()
    mean_diff = diff_tensor.mean().item()

    cos_sim = F.cosine_similarity(
        orig_trimmed.unsqueeze(0),
        adv_trimmed.unsqueeze(0)
    ).item()

    orig_clipped = torch.clamp(orig_trimmed, min=1e-10)
    adv_clipped = torch.clamp(adv_trimmed, min=1e-10)
    kl_div = (orig_clipped * torch.log(orig_clipped / adv_clipped)).sum().item()

    m = 0.5 * (orig_clipped + adv_clipped)
    js_div = 0.5 * (
        (orig_clipped * torch.log(orig_clipped / m)).sum().item() +
        (adv_clipped * torch.log(adv_clipped / m)).sum().item()
    )

    orig_np = orig_trimmed.cpu().numpy()
    adv_np = adv_trimmed.cpu().numpy()
    corr = np.corrcoef(orig_np, adv_np)[0, 1]
    if np.isnan(corr):
        corr = 0.0

    method_diffs = {
        'entropy': entropy_info['difference'],
        'permutation': perm_info['difference'],
        'random': rand_info['difference']
    }

    # top-5 original attention word indices
    top5_orig_indices = torch.argsort(original_attention[:valid_len], descending=True)[:5].cpu().tolist()
    top5_adv_indices = torch.argsort(best_adv_attention[:valid_len], descending=True)[:5].cpu().tolist()

    top5_overlap = len(set(top5_orig_indices) & set(top5_adv_indices))

    prediction_diff = abs(original_prediction - adv_prediction)
    same_class = (original_prediction >= 0.5) == (adv_prediction >= 0.5)

    return {
        'original_prediction': float(original_prediction),
        'adversarial_prediction': float(adv_prediction),
        'prediction_difference': float(prediction_diff),
        'same_class': bool(same_class),
        'best_method': best_method,
        'l1_difference': float(l1_diff),
        'l2_difference': float(l2_diff),
        'max_difference': float(max_diff),
        'mean_difference': float(mean_diff),
        'cosine_similarity': float(cos_sim),
        'kl_divergence': float(kl_div),
        'js_divergence': float(js_div),
        'pearson_correlation': float(corr),
        'method_differences': method_diffs,
        'top5_original_indices': top5_orig_indices,
        'top5_adversarial_indices': top5_adv_indices,
        'top5_overlap': top5_overlap,
        'valid_length': valid_len,
        'original_attention': original_attention[:valid_len].cpu().tolist(),
        'adversarial_attention': best_adv_attention[:valid_len].cpu().tolist(),
    }


def load_model(device: torch.device) -> AttentionClassifier:

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

    model_path = PROJECT_ROOT / "checkpoints" / "bilstm_model.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    model = load_model(device)

    preprocessor = Preprocessor.from_vocab_file(
        Config.VOCAB_FILE,
        max_length=Config.MAX_SEQ_LENGTH
    )
    test_dataset = IMDBDataset(
        str(Config.DATA_DIR),
        preprocessor,
        split="test"
    )

    all_results = []
    start_time = time.time()

    for idx in tqdm(range(len(test_dataset)), desc="Adversarial Attack"):
        token_ids, label, length = test_dataset[idx]

        token_ids = token_ids.unsqueeze(0).to(device)
        lengths = torch.tensor([length], dtype=torch.long).to(device)

        try:
            result = run_attack_single_sample(model, token_ids, lengths, device)
            result['sample_index'] = idx
            result['true_label'] = int(label.item())
            all_results.append(result)
        except Exception as e:
            print(f"\n  Warning: Sample {idx} failed: {e}")
            continue

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            eta = (len(test_dataset) - idx - 1) / rate
            print(f"\n  [{idx+1}/{len(test_dataset)}] "
                  f"Rate: {rate:.1f} samples/sec | "
                  f"ETA: {eta/60:.1f} min")

            checkpoint_path = results_dir / "full_test_results_checkpoint.json"
            with open(checkpoint_path, 'w') as f:
                json.dump({'results': all_results, 'completed': idx + 1}, f)

    total_time = time.time() - start_time

    final_output = {
        'experiment': 'adversarial_attention_attack',
        'dataset': 'IMDB Test Set',
        'total_samples': len(all_results),
        'total_time_seconds': total_time,
        'device': str(device),
        'results': all_results
    }

    output_path = results_dir / "full_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"Results saved to: {output_path}")

    l1_diffs = [r['l1_difference'] for r in all_results]
    cos_sims = [r['cosine_similarity'] for r in all_results]
    pred_diffs = [r['prediction_difference'] for r in all_results]
    same_class_count = sum(1 for r in all_results if r['same_class'])
    method_counts = {}
    for r in all_results:
        m = r['best_method']
        method_counts[m] = method_counts.get(m, 0) + 1
    top5_overlaps = [r['top5_overlap'] for r in all_results]
    kl_divs = [r['kl_divergence'] for r in all_results]
    js_divs = [r['js_divergence'] for r in all_results]
    correlations = [r['pearson_correlation'] for r in all_results]

    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method:15s}: {count:5d} ({100*count/len(all_results):.1f}%)")

    summary = {
        'total_samples': len(all_results),
        'total_time_seconds': total_time,
        'avg_l1_difference': float(np.mean(l1_diffs)),
        'std_l1_difference': float(np.std(l1_diffs)),
        'avg_cosine_similarity': float(np.mean(cos_sims)),
        'std_cosine_similarity': float(np.std(cos_sims)),
        'avg_kl_divergence': float(np.mean(kl_divs)),
        'std_kl_divergence': float(np.std(kl_divs)),
        'avg_js_divergence': float(np.mean(js_divs)),
        'std_js_divergence': float(np.std(js_divs)),
        'avg_pearson_correlation': float(np.mean(correlations)),
        'std_pearson_correlation': float(np.std(correlations)),
        'avg_prediction_difference': float(np.mean(pred_diffs)),
        'std_prediction_difference': float(np.std(pred_diffs)),
        'same_class_rate': float(same_class_count / len(all_results)),
        'same_class_count': same_class_count,
        'method_distribution': method_counts,
        'avg_top5_overlap': float(np.mean(top5_overlaps)),
    }
    summary_path = results_dir / "summary_statistics.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    checkpoint_path = results_dir / "full_test_results_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == '__main__':
    main()
