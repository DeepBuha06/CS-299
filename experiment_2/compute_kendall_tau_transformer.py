import torch
import torch.nn.functional as F
import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.stats import kendalltau
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_transformer import TransformerConfig
from models_transformer.model import TransformerClassifier
from models_transformer.dataset import IMDBTransformerDataset, get_tokenizer


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


def compute_gradient_and_attention(model, input_ids, attention_mask, device):
    
    model.eval()
    model.zero_grad()

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    captured_embeddings = {}

    def embed_hook(module, input, output):
        # gradients w.r.t. the embedding output
        captured_embeddings['value'] = output
        output.retain_grad()
        return output

    handle = model.distilbert.embeddings.register_forward_hook(embed_hook)

    # forward pass
    outputs = model.distilbert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True
    )

    handle.remove()

    # cls classification
    cls_output = outputs.last_hidden_state[:, 0, :]
    logits = model.classifier(model.dropout(cls_output))
    probs = torch.softmax(logits, dim=-1)
    pred_class = torch.argmax(probs, dim=-1).item()
    pred_prob = probs[0, pred_class]

    # backward
    pred_prob.backward()

    # gradient importance from captured embeddings
    emb = captured_embeddings['value']
    if emb.grad is None:
        raise RuntimeError("No gradient on embeddings")
    grad = emb.grad[0]  # (seq_len, hidden_dim)
    valid_len = int(attention_mask[0].sum().item())
    gradient_importance = torch.norm(grad[:valid_len], dim=1)
    grad_sum = gradient_importance.sum()
    if grad_sum > 0:
        gradient_importance = gradient_importance / grad_sum

    # last layer attention
    last_attn = outputs.attentions[-1]  # (1, num_heads, seq_len, seq_len)
    avg_attn = last_attn.mean(dim=1)  # (1, seq_len, seq_len)
    cls_attention = avg_attn[0, 0, :valid_len]

    attn_sum = cls_attention.sum()
    if attn_sum > 0:
        cls_attention = cls_attention / attn_sum

    return gradient_importance.detach().cpu().numpy(), cls_attention.detach().cpu().numpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results_dir = Path(__file__).parent / "results_transformer"
    results_dir.mkdir(exist_ok=True)

    print("Loading DistilBERT model...")
    model = load_model(device)

    print("Loading IMDB test dataset...")
    tokenizer = get_tokenizer(TransformerConfig.MODEL_NAME)
    test_dataset = IMDBTransformerDataset(
        str(TransformerConfig.DATA_DIR),
        tokenizer,
        max_length=TransformerConfig.MAX_SEQ_LENGTH,
        split="test"
    )

    kendall_taus = []
    start_time = time.time()

    for idx in tqdm(range(len(test_dataset)), desc="Kendall Tau (Transformer)"):
        sample = test_dataset[idx]
        input_ids = sample['input_ids'].unsqueeze(0)
        attention_mask = sample['attention_mask'].unsqueeze(0)

        try:
            grad_imp, attn = compute_gradient_and_attention(model, input_ids, attention_mask, device)

            if len(grad_imp) > 1:
                tau, _ = kendalltau(attn, grad_imp)
                if np.isnan(tau):
                    tau = 0.0
            else:
                tau = 0.0

            kendall_taus.append(float(tau))
        except Exception as e:
            print(f"\n  Warning: Sample {idx} failed: {e}")
            kendall_taus.append(0.0)

        if (idx + 1) % 500 == 0:
            torch.cuda.empty_cache() if device.type == 'cuda' else None

    total_time = time.time() - start_time

    output = {
        'kendall_taus': kendall_taus,
        'total_samples': len(kendall_taus),
        'total_time_seconds': total_time,
        'avg_kendall_tau': float(np.mean(kendall_taus)),
        'std_kendall_tau': float(np.std(kendall_taus)),
        'median_kendall_tau': float(np.median(kendall_taus)),
    }
    with open(results_dir / "kendall_tau_transformer.json", 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
