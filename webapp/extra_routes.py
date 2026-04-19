
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
from extra.integrated_gradients.ig_sentiment_preservation import AttentionReplacer, make_ig_attention_matrix, get_raw_attention_distribution
import json
from extra.relevance_propagation.lrp import compute_relevance_map
from extra.integrated_gradients.ig import compute_integrated_gradients, pool_subword_attributions
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
    n = len(raw)
    if n < 2:
        return {'kendall_tau': 0, 'rank_changes': [], 'top5_raw': [], 'top5_rollout': []}

    raw_ranks = np.argsort(np.argsort(-np.array(raw)))  # 0 = highest
    roll_ranks = np.argsort(np.argsort(-np.array(rollout)))

    tau, _ = kendalltau(raw, rollout)
    if np.isnan(tau):
        tau = 0.0

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

    raw_attn = attentions[-1].mean(dim=1)[0, 0, :valid_len].cpu().numpy()
    rs = raw_attn.sum()
    if rs > 0:
        raw_attn = raw_attn / rs

    cls_rel = get_cls_rollout(attentions, discard_ratio=0.1)[0][:valid_len].cpu().numpy()
    rls = cls_rel.sum()
    if rls > 0:
        cls_rel = cls_rel / rls

    tokens = _tokenizer.convert_ids_to_tokens(input_ids[0])[:valid_len]
    tokens_clean, raw_clean, roll_clean = _filter_special(tokens, raw_attn.tolist(), cls_rel.tolist())

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

@extra_bp.route('/ig_aggregate', methods=['GET'])
def ig_aggregate():
    project_root = Path(__file__).parent.parent
    result_path = project_root / 'extra' / 'integrated_gradients' / 'ig_preservation_results.json'
    if result_path.exists():
        with open(result_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Results not found.'}), 404

def _ig_filter_special(tokens, attributions):
    import string
    PUNCT = set(string.punctuation)
    idx = [i for i, t in enumerate(tokens) if t not in SPECIAL_TOKENS and t not in PUNCT]
    toks = [tokens[i] for i in idx]
    atts = [attributions[i] for i in idx]
    
    abs_sum = sum(abs(a) for a in atts)
    if abs_sum > 0:
        atts = [a / abs_sum for a in atts]
        
    return toks, atts

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
    
    attribution_full, _ = compute_integrated_gradients(_model, input_ids, attention_mask, target_class=target, steps=50)
    
    seq_len = input_ids.shape[1]
    attn_mask_np = attention_mask[0].cpu().numpy().astype(np.float32)
    
    with torch.no_grad():
        raw_attn = get_raw_attention_distribution(_model, input_ids, attention_mask, valid_len)
        
        ig_normalized = make_ig_attention_matrix(attribution_full, seq_len, attn_mask_np)

        ig_valid = ig_normalized[:valid_len].numpy()
        ig_valid_sum = ig_valid.sum()
        if ig_valid_sum > 1e-9:
            ig_valid = ig_valid / ig_valid_sum
            
        tau = 0.0
        if len(ig_valid) > 1:
            tau, _ = kendalltau(raw_attn, ig_valid)
            if np.isnan(tau): tau = 0.0

        replacer = AttentionReplacer(_model, ig_normalized)
        replacer.attach()
        try:
            logits_ig, _ = _model(input_ids, attention_mask)
            probs_ig = torch.softmax(logits_ig, dim=-1)
            pred_ig = torch.argmax(probs_ig, dim=-1).item()
            conf_ig = probs_ig[0, pred_ig].item()
        finally:
            replacer.remove()

    flipped = pred_class != pred_ig
    prob_shift = abs(probs[0, pred_class].item() - probs_ig[0, pred_class].item())

    attribution = attribution_full[:valid_len]
    tokens = _tokenizer.convert_ids_to_tokens(input_ids[0])[:valid_len]
    pooled_tokens, pooled_attrs = pool_subword_attributions(tokens, attribution.tolist())
    _, pooled_raw = pool_subword_attributions(tokens, raw_attn.tolist())
    _, pooled_ig_dist = pool_subword_attributions(tokens, ig_valid.tolist())
    
    tokens_clean, ig_clean = _ig_filter_special(pooled_tokens, pooled_attrs.tolist())
    
    _, raw_dist_clean = _ig_filter_special(pooled_tokens, pooled_raw.tolist())
    _, ig_dist_clean = _ig_filter_special(pooled_tokens, pooled_ig_dist.tolist())

    return jsonify({
        'prediction': 'Positive' if pred_class == 1 else 'Negative',
        'confidence': float(probs[0, pred_class].item()) * 100,
        'target_class': target,
        'tokens': tokens_clean,
        'ig_attribution': ig_clean,
        'raw_distribution': raw_dist_clean,
        'ig_distribution': ig_dist_clean,
        'faithfulness': {
            'pred_ig': 'Positive' if pred_ig == 1 else 'Negative',
            'conf_ig': float(conf_ig) * 100,
            'flipped': flipped,
            'prob_shift': float(prob_shift) * 100,
            'kendall_tau': float(tau)
        }
    })
