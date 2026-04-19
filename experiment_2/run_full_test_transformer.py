import torch
import torch.nn.functional as F
import json
import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_transformer import TransformerConfig
from models_transformer.model import TransformerClassifier
from models_transformer.dataset import IMDBTransformerDataset, get_tokenizer


def get_attention_and_prediction(model, input_ids, attention_mask, device):
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits, cls_attention = model(input_ids, attention_mask, return_attention=True)
        probs = torch.softmax(logits, dim=-1)

    # cls_attention
    attn = cls_attention[0]  # (seq_len,)
    pred_class = torch.argmax(probs, dim=-1).item()
    pred_prob = probs[0, pred_class].item()

    # hidden states for adversarial re-scoring
    with torch.no_grad():
        outputs = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

    return attn, pred_class, pred_prob, probs[0], hidden_states


def adversarial_entropy(original_attention, attention_mask, device):
    seq_len = original_attention.shape[0]
    valid_len = int(attention_mask[0].sum().item())
    uniform = torch.zeros(seq_len, device=device)
    if valid_len > 0:
        uniform[:valid_len] = 1.0 / valid_len
    diff = torch.abs(uniform - original_attention).sum().item()
    return uniform, {'method': 'entropy', 'difference': diff}


def adversarial_permutation(original_attention, attention_mask, device, num_permutations=100):
    valid_len = int(attention_mask[0].sum().item())
    seq_len = original_attention.shape[0]
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

        perm_tensor = torch.zeros(seq_len, device=device)
        perm_tensor[:valid_len] = torch.tensor(perm, device=device, dtype=torch.float32)

        diff = torch.abs(perm_tensor - original_attention).sum().item()
        if diff > best_diff:
            best_diff = diff
            best_attention = perm_tensor.clone()

    if best_attention is None:
        best_attention = original_attention.clone()

    return best_attention, {'method': 'permutation', 'difference': best_diff}


def adversarial_random(original_attention, attention_mask, device, num_samples=500):
    valid_len = int(attention_mask[0].sum().item())
    seq_len = original_attention.shape[0]
    best_attention = None
    best_diff = 0.0

    for _ in range(num_samples):
        rand_attn = torch.zeros(seq_len, device=device)
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


def compute_adversarial_prediction_transformer(model, hidden_states, adv_attention, device):
    model.eval()
    with torch.no_grad():
        # adv_attention
        adv_attn_batch = adv_attention.unsqueeze(0).unsqueeze(1)
        # hidden_states
        context = torch.bmm(adv_attn_batch, hidden_states).squeeze(1)  # (1, hidden_dim)
        
        pooled_output = model.dropout(context)
        logits = model.classifier(pooled_output)
        probs = torch.softmax(logits, dim=-1)

    pred_class = torch.argmax(probs, dim=-1).item()
    pred_prob = probs[0, pred_class].item()
    return pred_class, pred_prob, probs[0]


def run_attack_single_sample_transformer(model, input_ids, attention_mask, device):
    
    attn, orig_class, orig_prob, orig_probs, hidden_states = get_attention_and_prediction(
        model, input_ids, attention_mask, device
    )

    valid_len = int(attention_mask[0].sum().item())
    seq_len = attn.shape[0]

    entropy_attn, entropy_info = adversarial_entropy(attn, attention_mask, device)
    perm_attn, perm_info = adversarial_permutation(attn, attention_mask, device, num_permutations=100)
    rand_attn, rand_info = adversarial_random(attn, attention_mask, device, num_samples=500)

    methods = {
        'entropy': (entropy_attn, entropy_info),
        'permutation': (perm_attn, perm_info),
        'random': (rand_attn, rand_info)
    }
    best_method = max(methods.keys(), key=lambda k: methods[k][1]['difference'])
    best_adv_attention = methods[best_method][0]
    best_diff = methods[best_method][1]['difference']

    # adversarial prediction
    adv_class, adv_prob, adv_probs = compute_adversarial_prediction_transformer(
        model, hidden_states, best_adv_attention, device
    )

    # metrics
    orig_trimmed = attn[:valid_len]
    adv_trimmed = best_adv_attention[:valid_len]

    diff_tensor = torch.abs(orig_trimmed - adv_trimmed)
    l1_diff = diff_tensor.sum().item()
    l2_diff = torch.sqrt((diff_tensor ** 2).sum()).item()
    max_diff = diff_tensor.max().item() if valid_len > 0 else 0.0
    mean_diff = diff_tensor.mean().item() if valid_len > 0 else 0.0

    cos_sim = F.cosine_similarity(orig_trimmed.unsqueeze(0), adv_trimmed.unsqueeze(0)).item() if valid_len > 0 else 0.0

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
    corr = np.corrcoef(orig_np, adv_np)[0, 1] if valid_len > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0

    method_diffs = {
        'entropy': entropy_info['difference'],
        'permutation': perm_info['difference'],
        'random': rand_info['difference']
    }

    top5_orig = torch.argsort(orig_trimmed, descending=True)[:5].cpu().tolist() if valid_len > 0 else []
    top5_adv = torch.argsort(adv_trimmed, descending=True)[:5].cpu().tolist() if valid_len > 0 else []
    top5_overlap = len(set(top5_orig) & set(top5_adv))

    same_class = (orig_class == adv_class)

    return {
        'original_prediction_class': orig_class,
        'original_prediction_prob': float(orig_prob),
        'adversarial_prediction_class': adv_class,
        'adversarial_prediction_prob': float(adv_prob),
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
        'top5_overlap': top5_overlap,
        'valid_length': valid_len,
    }


def load_model(device):
    model = TransformerClassifier(
        model_name=TransformerConfig.MODEL_NAME,
        num_labels=TransformerConfig.NUM_LABELS
    )
    model_path = PROJECT_ROOT / "checkpoints" / TransformerConfig.MODEL_CHECKPOINT
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results_dir = Path(__file__).parent / "results_transformer"
    results_dir.mkdir(exist_ok=True)

    model = load_model(device)

    tokenizer = get_tokenizer(TransformerConfig.MODEL_NAME)
    test_dataset = IMDBTransformerDataset(
        str(TransformerConfig.DATA_DIR),
        tokenizer,
        max_length=TransformerConfig.MAX_SEQ_LENGTH,
        split="test"
    )
    
    all_results = []
    start_time = time.time()

    for idx in tqdm(range(len(test_dataset)), desc="Transformer Adversarial"):
        sample = test_dataset[idx]
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        label = sample['label'].item()

        try:
            result = run_attack_single_sample_transformer(model, input_ids, attention_mask, device)
            result['sample_index'] = idx
            result['true_label'] = label
            all_results.append(result)
        except Exception as e:
            print(f"\n  Warning: Sample {idx} failed: {e}")
            continue

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            eta = (len(test_dataset) - idx - 1) / rate
            print(f"\n  [{idx+1}/{len(test_dataset)}] "
                  f"Rate: {rate:.1f} samples/sec | ETA: {eta/60:.1f} min")

            cp = results_dir / "transformer_results_checkpoint.json"
            with open(cp, 'w') as f:
                json.dump({'results': all_results, 'completed': idx + 1}, f)

    total_time = time.time() - start_time

    final = {
        'experiment': 'adversarial_attention_attack_transformer',
        'model': 'DistilBERT',
        'dataset': 'IMDB Test Set',
        'total_samples': len(all_results),
        'total_time_seconds': total_time,
        'device': str(device),
        'results': all_results
    }
    out_path = results_dir / "full_test_results_transformer.json"
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2)

    l1_diffs = [r['l1_difference'] for r in all_results]
    cos_sims = [r['cosine_similarity'] for r in all_results]
    js_divs = [r['js_divergence'] for r in all_results]
    same_count = sum(1 for r in all_results if r['same_class'])
    method_counts = {}
    for r in all_results:
        m = r['best_method']
        method_counts[m] = method_counts.get(m, 0) + 1

    summary = {
        'total_samples': len(all_results),
        'total_time_seconds': total_time,
        'avg_l1_difference': float(np.mean(l1_diffs)),
        'std_l1_difference': float(np.std(l1_diffs)),
        'avg_cosine_similarity': float(np.mean(cos_sims)),
        'avg_js_divergence': float(np.mean(js_divs)),
        'same_class_rate': float(same_count / len(all_results)),
        'same_class_count': same_count,
        'method_distribution': method_counts,
    }
    with open(results_dir / "summary_statistics_transformer.json", 'w') as f:
        json.dump(summary, f, indent=2)

    cp = results_dir / "transformer_results_checkpoint.json"
    if cp.exists():
        cp.unlink()
        

if __name__ == '__main__':
    main()
