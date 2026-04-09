"""
Flask routes for Extra Tasks: Attention Rollout & Relevance Propagation.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Blueprint, request, jsonify
import torch

from config_transformer import TransformerConfig
from models_transformer.model import TransformerClassifier
from models_transformer.dataset import get_tokenizer
from extra.attention_rollout.rollout import get_cls_rollout
from extra.relevance_propagation.lrp import compute_relevance_map
from extra.integrated_gradients.ig import compute_integrated_gradients

extra_bp = Blueprint('extra', __name__, url_prefix='/extra')

_model = None
_tokenizer = None
_initialized = False

SPECIAL_TOKENS = {'[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'}


def _init():
    global _model, _tokenizer, _initialized
    if _initialized:
        return True
    project_root = Path(__file__).parent.parent
    try:
        _tokenizer = get_tokenizer(TransformerConfig.MODEL_NAME)
        _model = TransformerClassifier(
            model_name=TransformerConfig.MODEL_NAME,
            num_labels=TransformerConfig.NUM_LABELS
        )
        model_path = project_root / 'checkpoints' / TransformerConfig.MODEL_CHECKPOINT
        if not model_path.exists():
            return False
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            _model.load_state_dict(checkpoint['model_state_dict'])
        else:
            _model.load_state_dict(checkpoint)
        _model.eval()
        _initialized = True
        return True
    except Exception as e:
        print(f"Extra init failed: {e}")
        return False


def _tokenize_text(text):
    encoding = _tokenizer(text, truncation=True, max_length=TransformerConfig.MAX_SEQ_LENGTH,
                          padding='max_length', return_tensors='pt')
    return encoding['input_ids'], encoding['attention_mask']


import string

def _filter_special(tokens, *arrs):
    """Remove special tokens and structural punctuation, re-normalize score arrays."""
    # BERT uses special tokens and punctuation (like '.') as structural hubs/aggregators.
    # We filter them out so the visualizations highlight semantic content words.
    PUNCT = set(string.punctuation)
    
    idx = [i for i, t in enumerate(tokens) if t not in SPECIAL_TOKENS and t not in PUNCT]
    toks = [tokens[i] for i in idx]
    out = []
    for arr in arrs:
        vals = [arr[i] for i in idx]
        s = sum(vals)
        if s > 0:
            vals = [v / s for v in vals]
        out.append(vals)
    return toks, *out


def _rank_diff_stats(raw, rollout):
    """Compute rank-based difference metrics between raw attention and rollout."""
    n = len(raw)
    if n < 2:
        return {'kendall_tau': 0, 'rank_changes': [], 'top5_raw': [], 'top5_rollout': []}

    raw_ranks = np.argsort(np.argsort(-np.array(raw)))  # 0 = highest
    roll_ranks = np.argsort(np.argsort(-np.array(rollout)))

    tau, _ = kendalltau(raw, rollout)
    if np.isnan(tau):
        tau = 0.0

    # Rank changes: positive = promoted by rollout, negative = demoted
    rank_changes = (raw_ranks - roll_ranks).tolist()

    top5_raw = np.argsort(-np.array(raw))[:5].tolist()
    top5_rollout = np.argsort(-np.array(rollout))[:5].tolist()

    return {
        'kendall_tau': float(tau),
        'rank_changes': rank_changes,
        'top5_raw_idx': top5_raw,
        'top5_rollout_idx': top5_rollout,
    }


# ============ TASK 1: ATTENTION ROLLOUT ============

@extra_bp.route('/rollout', methods=['POST'])
def rollout_analysis():
    if not _init():
        return jsonify({'error': 'Model not loaded'}), 500
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    input_ids, attention_mask = _tokenize_text(text)
    valid_len = int(attention_mask[0].sum().item())

    with torch.no_grad():
        logits, _ = _model(input_ids, attention_mask, return_attention=True)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()

        outputs = _model.distilbert(input_ids=input_ids, attention_mask=attention_mask,
                                     output_attentions=True)
        attentions = list(outputs.attentions)

    # Raw last-layer CLS attention (content tokens only for fair comparison)
    raw_attn = attentions[-1].mean(dim=1)[0, 0, :valid_len].cpu().numpy()
    rs = raw_attn.sum()
    if rs > 0:
        raw_attn = raw_attn / rs

    # Rollout with discard_ratio to sharpen signal
    cls_rel = get_cls_rollout(attentions, discard_ratio=0.1)[0][:valid_len].cpu().numpy()
    rls = cls_rel.sum()
    if rls > 0:
        cls_rel = cls_rel / rls

    tokens = _tokenizer.convert_ids_to_tokens(input_ids[0])[:valid_len]
    tokens_clean, raw_clean, roll_clean = _filter_special(tokens, raw_attn.tolist(), cls_rel.tolist())

    # Rank analysis
    stats = _rank_diff_stats(raw_clean, roll_clean)

    return jsonify({
        'prediction': 'Positive' if pred_class == 1 else 'Negative',
        'confidence': float(probs[0, pred_class].item()) * 100,
        'tokens': tokens_clean,
        'raw_attention': raw_clean,
        'rollout_relevance': roll_clean,
        'rank_stats': stats,
    })


# ============ TASK 2: RELEVANCE PROPAGATION (LRP) ============

@extra_bp.route('/lrp', methods=['POST'])
def lrp_analysis():
    if not _init():
        return jsonify({'error': 'Model not loaded'}), 500
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    input_ids, attention_mask = _tokenize_text(text)
    valid_len = int(attention_mask[0].sum().item())

    with torch.no_grad():
        logits, _ = _model(input_ids, attention_mask, return_attention=True)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()

    # Raw attention for comparison
    with torch.no_grad():
        outputs = _model.distilbert(input_ids=input_ids, attention_mask=attention_mask,
                                     output_attentions=True)
        raw_attn = outputs.attentions[-1].mean(dim=1)[0, 0, :valid_len].cpu().numpy()
        rs = raw_attn.sum()
        if rs > 0:
            raw_attn = raw_attn / rs

    target = data.get('target_class', pred_class)
    relevance = compute_relevance_map(_model, input_ids, attention_mask, target_class=target)
    relevance = relevance[:valid_len]

    tokens = _tokenizer.convert_ids_to_tokens(input_ids[0])[:valid_len]
    tokens_clean, raw_clean, lrp_clean = _filter_special(tokens, raw_attn.tolist(), relevance.tolist())

    stats = _rank_diff_stats(raw_clean, lrp_clean)

    return jsonify({
        'prediction': 'Positive' if pred_class == 1 else 'Negative',
        'confidence': float(probs[0, pred_class].item()) * 100,
        'target_class': target,
        'tokens': tokens_clean,
        'raw_attention': raw_clean,
        'lrp_relevance': lrp_clean,
        'rank_stats': stats,
    })

# ============ TASK 3: INTEGRATED GRADIENTS (INPUT ATTRIBUTION) ============

@extra_bp.route('/ig', methods=['POST'])
def ig_analysis():
    if not _init():
        return jsonify({'error': 'Model not loaded'}), 500
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    input_ids, attention_mask = _tokenize_text(text)
    valid_len = int(attention_mask[0].sum().item())

    with torch.no_grad():
        logits, _ = _model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()

    target = data.get('target_class', pred_class)
    
    # Compute Integrated Gradients
    attribution, _ = compute_integrated_gradients(_model, input_ids, attention_mask, target_class=target, steps=50)
    attribution = attribution[:valid_len]

    tokens = _tokenizer.convert_ids_to_tokens(input_ids[0])[:valid_len]
    
    # We use _filter_special to remove structural punctuation/special tokens for final clean visualization
    # We pass attribution twice just to satisfy the *arrs unpacker structure
    tokens_clean, ig_clean, _ = _filter_special(tokens, attribution.tolist(), attribution.tolist())

    return jsonify({
        'prediction': 'Positive' if pred_class == 1 else 'Negative',
        'confidence': float(probs[0, pred_class].item()) * 100,
        'target_class': target,
        'tokens': tokens_clean,
        'ig_attribution': ig_clean
    })
