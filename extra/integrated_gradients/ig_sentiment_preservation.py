"""
Sentiment Preservation Test: Using IG Attributions as Attention Weights.

This script tests whether Integrated Gradient attributions can serve as a
faithful replacement for the model's learned attention weights.

The idea (from Jain & Wallace 2019, "Attention is not Explanation"):
    If we replace the model's internal attention distributions with
    IG-derived distributions, does the model's prediction stay the same?

Setup:
    1. Load test samples from IMDB
    2. For each sample, run IG to get per-token attribution scores
    3. Normalize IG scores to form a valid probability distribution
    4. Hook into every transformer layer and REPLACE the attention weights
       with the IG distribution (broadcast across heads and query positions)
    5. Compare:  original_prediction vs ig_attention_prediction

Metrics:
    - Decision Flip Rate: how often does the sentiment change?
    - Probability Shift: how much does the confidence change on average?
    - Kendall-tau: rank correlation between original attention and IG weights

Usage:
    python ig_sentiment_preservation.py [--num_samples 200] [--steps 50]

Data: Uses IMDB test set (data/raw/imdb/test/pos and test/neg)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from scipy.stats import kendalltau

from config_transformer import TransformerConfig
from models_transformer.model import TransformerClassifier
from models_transformer.dataset import get_tokenizer
from extra.integrated_gradients.ig import compute_integrated_gradients


def load_model():
    tokenizer = get_tokenizer(TransformerConfig.MODEL_NAME)
    model = TransformerClassifier(
        model_name=TransformerConfig.MODEL_NAME,
        num_labels=TransformerConfig.NUM_LABELS
    )
    model_path = PROJECT_ROOT / 'checkpoints' / TransformerConfig.MODEL_CHECKPOINT
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, tokenizer


def load_test_samples(data_dir, num_samples=200):
    samples = []
    per_class = num_samples // 2
    for label_name, label_id in [("neg", 0), ("pos", 1)]:
        folder = Path(data_dir) / "test" / label_name
        if not folder.exists():
            print(f"WARNING: {folder} not found!")
            continue
        files = sorted(folder.glob("*.txt"))[:per_class]
        for f in files:
            text = f.read_text(encoding='utf-8').strip()
            samples.append((text, label_id))
    return samples


def make_ig_attention_matrix(ig_scores_raw, seq_len, attention_mask_1d):
    """
    Convert raw IG token scores into a valid attention matrix.

    Shape needed by DistilBERT attention:
        (batch=1, num_heads, seq_len, seq_len)

    What we do:
        - Take abs() of IG scores so negative attribution still counts as
          "important" — attention weights must be non-negative
        - Zero out PAD positions using the attention mask
        - Normalize each ROW to sum to 1 (each query position needs a
          valid probability distribution over keys)
        - Broadcast the same key-distribution to every query position
          (this is the fairest single-vector → matrix expansion)

    FIX vs original code:
        Original did ig_d[:seq_len].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        and expanded to (batch, heads, seq_len, seq_len) — this is correct
        in shape but the normalization must happen AFTER masking PAD tokens,
        not before. Original normalized before masking, so PAD positions
        got small but nonzero weight, leaking signal from padding.
    """
    # ig_scores_raw is a numpy array of shape (full_padded_seq_len,)
    # Trim or pad to exactly seq_len
    scores = np.zeros(seq_len, dtype=np.float32)
    copy_len = min(len(ig_scores_raw), seq_len)
    scores[:copy_len] = ig_scores_raw[:copy_len]

    # Take absolute value — attention weights must be >= 0
    scores = np.abs(scores)

    # Zero out PAD token positions using attention mask
    # attention_mask_1d is a 1D numpy array of 0s and 1s, shape (seq_len,)
    scores = scores * attention_mask_1d

    # Normalize to sum = 1 (if all zero, use uniform over non-PAD positions)
    total = scores.sum()
    if total > 1e-9:
        scores = scores / total
    else:
        # Fallback: uniform over non-PAD tokens
        non_pad = attention_mask_1d.sum()
        scores = attention_mask_1d / max(non_pad, 1)

    # Shape: (seq_len,) → (1, 1, 1, seq_len)
    # We will expand to (batch=1, num_heads, seq_len, seq_len) in the hook
    ig_tensor = torch.tensor(scores, dtype=torch.float32)
    return ig_tensor  # shape: (seq_len,)


class AttentionReplacer:
    """
    Hook-based attention replacer for DistilBERT.

    Replaces softmax attention weights in every transformer layer with a
    custom distribution derived from IG attributions.

    FIX vs original:
        1. Normalization of IG vector happens AFTER masking PADs (in
           make_ig_attention_matrix), not before.
        2. The ig_attn matrix is properly shaped as (B, H, S, S) where
           every query row is the same IG key-distribution. This is
           correct — we're saying "no matter which token is querying,
           look at keys according to IG importance."
        3. We do NOT use the original_forward at all — we recompute
           q, k, v from scratch which avoids any state leakage from
           the original attention computation path.
    """

    def __init__(self, model, ig_distribution):
        """
        Args:
            model: TransformerClassifier
            ig_distribution: (seq_len,) tensor — normalized IG scores,
                             already abs-valued and PAD-masked
        """
        self.model = model
        self.ig_dist = ig_distribution  # (seq_len,)
        self._original_forwards = []

    def attach(self):
        for layer in self.model.distilbert.transformer.layer:
            sa = layer.attention
            self._original_forwards.append(sa.forward)
            ig_dist = self.ig_dist

            def make_hooked_forward(sa_module, ig_d):
                def hooked_forward(hidden_states, attention_mask=None,
                                   head_mask=None, output_attentions=False,
                                   **kwargs):
                    batch_size, seq_len, _ = hidden_states.shape
                    num_heads = sa_module.n_heads
                    head_size = sa_module.attention_head_size

                    # Recompute Q, K, V projections exactly as DistilBERT does
                    def shape(x):
                        # (B, S, D) → (B, H, S, head_size)
                        return x.view(batch_size, seq_len, num_heads, head_size).transpose(1, 2)

                    query_layer = shape(sa_module.q_lin(hidden_states))
                    key_layer   = shape(sa_module.k_lin(hidden_states))
                    value_layer = shape(sa_module.v_lin(hidden_states))

                    # Build IG attention matrix: (1, 1, 1, seq_len)
                    # → expand to (B, H, seq_len, seq_len)
                    # Every query position attends to keys by IG importance
                    ig_key_dist = ig_d[:seq_len].to(hidden_states.device)
                    # Shape: (B, H, S, S) — same key-distribution for every query row
                    ig_attn = ig_key_dist.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    ig_attn = ig_attn.expand(batch_size, num_heads, seq_len, seq_len)
                    # Make contiguous so matmul works cleanly
                    ig_attn = ig_attn.contiguous()

                    # Weighted sum of values using IG attention
                    # (B, H, S, S) x (B, H, S, head_size) → (B, H, S, head_size)
                    context = torch.matmul(ig_attn, value_layer)

                    # Merge heads: (B, H, S, head_size) → (B, S, H*head_size)
                    context = context.transpose(1, 2).contiguous()
                    context = context.reshape(batch_size, seq_len, num_heads * head_size)

                    # Output projection
                    output = sa_module.out_lin(context)

                    # Always return a tuple of len 2 to satisfy DistilBERT's TransformerBlock 
                    # unpacking: `attention_output, _ = self.attention(...)`
                    return (output, ig_attn)
                return hooked_forward

            sa.forward = make_hooked_forward(sa, ig_dist)

    def remove(self):
        for layer, orig_fwd in zip(
            self.model.distilbert.transformer.layer,
            self._original_forwards
        ):
            layer.attention.forward = orig_fwd
        self._original_forwards = []


def get_raw_attention_distribution(model, input_ids, attention_mask, valid_len):
    """
    Get the model's natural attention distribution from the last layer,
    averaged across all heads AND all query positions (not just [CLS]).

    FIX vs original:
        Original used [0, 0] which is only the [CLS] query row of the
        last layer's first head. This is biased.
        Correct: average across all heads, then average across all
        non-PAD query positions to get a fair key-distribution.
    """
    with torch.no_grad():
        outputs = model.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
    # Last layer attention: (batch=1, num_heads, seq_len, seq_len)
    last_layer_attn = outputs.attentions[-1][0]  # (H, S, S)

    # Average across heads → (S, S)
    avg_heads = last_layer_attn.mean(dim=0).cpu().numpy()

    # Average across non-PAD query positions → (S,)
    # Only rows 0..valid_len are real tokens
    avg_over_queries = avg_heads[:valid_len, :valid_len].mean(axis=0)  # (valid_len,)

    # Normalize
    total = avg_over_queries.sum()
    if total > 1e-9:
        avg_over_queries = avg_over_queries / total
    return avg_over_queries  # shape: (valid_len,)


def run_preservation_test(model, tokenizer, samples, steps=50, max_seq_length=256):
    results = {
        'total': 0,
        'flips': 0,
        'same_prediction': 0,
        'avg_prob_shift': [],
        'avg_kendall_tau': [],
        'per_sample': []
    }

    for text, true_label in tqdm(samples, desc="Testing IG sentiment preservation"):
        encoding = tokenizer(
            text, truncation=True, max_length=max_seq_length,
            padding='max_length', return_tensors='pt'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        valid_len = int(attention_mask[0].sum().item())
        seq_len = input_ids.shape[1]

        # 1. Original prediction (normal attention)
        with torch.no_grad():
            logits_orig, _ = model(input_ids, attention_mask)
            probs_orig = torch.softmax(logits_orig, dim=-1)
            pred_orig = torch.argmax(probs_orig, dim=-1).item()
            conf_orig = probs_orig[0, pred_orig].item()

        # 2. Compute IG attributions — returns (seq_len_with_special_tokens,)
        ig_attrs, target_class = compute_integrated_gradients(
            model, input_ids, attention_mask,
            target_class=pred_orig, steps=steps
        )

        # 3. Build proper IG attention distribution
        #    - abs values, PAD-masked, row-normalized
        attn_mask_np = attention_mask[0].cpu().numpy().astype(np.float32)
        ig_normalized = make_ig_attention_matrix(ig_attrs, seq_len, attn_mask_np)

        # 4. Get raw attention for Kendall-tau comparison
        raw_attn = get_raw_attention_distribution(
            model, input_ids, attention_mask, valid_len
        )

        # Kendall-tau: compare IG key-distribution vs raw attention key-distribution
        # Both trimmed to valid_len for a fair comparison
        ig_valid = ig_normalized[:valid_len].numpy()
        ig_valid_sum = ig_valid.sum()
        if ig_valid_sum > 1e-9:
            ig_valid = ig_valid / ig_valid_sum  # re-normalize after trimming
        raw_valid = raw_attn  # already valid_len

        if len(ig_valid) > 1:
            tau, _ = kendalltau(raw_valid, ig_valid)
            if np.isnan(tau):
                tau = 0.0
        else:
            tau = 0.0

        # 5. Replace attention with IG distribution and get new prediction
        replacer = AttentionReplacer(model, ig_normalized)
        replacer.attach()
        try:
            with torch.no_grad():
                logits_ig, _ = model(input_ids, attention_mask)
                probs_ig = torch.softmax(logits_ig, dim=-1)
                pred_ig = torch.argmax(probs_ig, dim=-1).item()
                conf_ig = probs_ig[0, pred_ig].item()
        finally:
            # Always restore original attention even if an error occurs
            replacer.remove()

        # 6. Record results
        flipped = pred_orig != pred_ig
        prob_shift = abs(probs_orig[0, 1].item() - probs_ig[0, 1].item())

        results['total'] += 1
        results['flips'] += int(flipped)
        results['same_prediction'] += int(not flipped)
        results['avg_prob_shift'].append(prob_shift)
        results['avg_kendall_tau'].append(tau)
        results['per_sample'].append({
            'true_label': true_label,
            'pred_original': pred_orig,
            'conf_original': round(conf_orig, 4),
            'pred_ig_attention': pred_ig,
            'conf_ig_attention': round(conf_ig, 4),
            'flipped': flipped,
            'prob_shift': round(prob_shift, 4),
            'kendall_tau': round(tau, 4),
        })

    results['flip_rate'] = results['flips'] / max(results['total'], 1)
    results['preservation_rate'] = results['same_prediction'] / max(results['total'], 1)
    results['mean_prob_shift'] = float(np.mean(results['avg_prob_shift']))
    results['mean_kendall_tau'] = float(np.mean(results['avg_kendall_tau']))

    del results['avg_prob_shift']
    del results['avg_kendall_tau']

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test if IG attributions can replace attention and preserve sentiment."
    )
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("IG SENTIMENT PRESERVATION TEST")
    print("=" * 60)
    print(f"Samples: {args.num_samples} | IG steps: {args.steps}")

    print("Loading model...")
    model, tokenizer = load_model()

    data_dir = PROJECT_ROOT / 'data' / 'raw' / 'imdb'
    print(f"Loading test samples from {data_dir}...")
    samples = load_test_samples(data_dir, args.num_samples)
    print(f"Loaded {len(samples)} samples")

    print("\nRunning preservation test...")
    results = run_preservation_test(
        model, tokenizer, samples,
        steps=args.steps,
        max_seq_length=TransformerConfig.MAX_SEQ_LENGTH
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total samples:          {results['total']}")
    print(f"Prediction preserved:   {results['same_prediction']} ({results['preservation_rate']:.1%})")
    print(f"Prediction flipped:     {results['flips']} ({results['flip_rate']:.1%})")
    print(f"Mean probability shift: {results['mean_prob_shift']:.4f}")
    print(f"Mean Kendall-tau:       {results['mean_kendall_tau']:.4f}")
    print()

    pr = results['preservation_rate']
    if pr > 0.8:
        print("INTERPRETATION: IG attributions largely PRESERVE sentiment.")
        print("  → IG captures the tokens the model actually relies on.")
    elif pr > 0.5:
        print("INTERPRETATION: IG attributions PARTIALLY preserve sentiment.")
        print("  → IG captures some but not all of the model's decision logic.")
    else:
        print("INTERPRETATION: IG attributions do NOT preserve sentiment well.")
        print("  → The model's internal attention routing carries information")
        print("    that pure input-level attribution cannot fully capture.")

    output_path = args.output or str(
        PROJECT_ROOT / 'extra' / 'integrated_gradients' / 'ig_preservation_results.json'
    )
    save_results = {k: v for k, v in results.items() if k != 'per_sample'}
    save_results['per_sample_count'] = len(results.get('per_sample', []))
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()