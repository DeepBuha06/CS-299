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

from config import Config
from models.model import AttentionClassifier
from data.preprocessing import Preprocessor
from data.dataset import IMDBDataset


def load_model(device):

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


def compute_gradient_importance(model, token_ids, lengths, device):
    
    model.train()
    
    # disable dropout layers
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    
    token_ids = token_ids.to(device)
    lengths = lengths.to(device)
    
    # zero existing gradients
    model.zero_grad()
    
    # enable gradient computation for embeddings
    embeddings = model.embedding(token_ids)  # (1, seq_len, embed_dim)
    embeddings = embeddings.detach().requires_grad_(True)
    
    # forward pass
    hidden_states, _ = model.encoder(embeddings, lengths)
    
    # attention
    batch_size, seq_length = token_ids.shape
    mask = None
    if lengths is not None:
        range_tensor = torch.arange(seq_length, device=device)
        range_tensor = range_tensor.unsqueeze(0).expand(batch_size, -1)
        mask = range_tensor < lengths.unsqueeze(1)
    
    context, attention_weights = model.attention(hidden_states, mask)
    
    prediction = model.classifier(context)
    
    # backward pass
    prediction.backward()
    
    # gradient importance
    grad = embeddings.grad[0]  # (seq_len, embed_dim)
    
    valid_len = int(lengths[0].item())
    gradient_importance = torch.norm(grad[:valid_len], dim=1)  # (valid_len,)
    
    # normalize
    grad_sum = gradient_importance.sum()
    if grad_sum > 0:
        gradient_importance = gradient_importance / grad_sum
    
    attn = attention_weights[0][:valid_len]  # (valid_len,)
    
    # model back to eval mode
    model.eval()
    
    return gradient_importance.detach().cpu().numpy(), attn.detach().cpu().numpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    kendall_taus = []
    kendall_pvalues = []
    start_time = time.time()

    for idx in tqdm(range(len(test_dataset)), desc="Kendall Tau"):
        token_ids, label, length = test_dataset[idx]
        token_ids = token_ids.unsqueeze(0).to(device)
        lengths = torch.tensor([length], dtype=torch.long).to(device)

        try:
            grad_imp, attn = compute_gradient_importance(model, token_ids, lengths, device)
            
            # compute Kendall Tau correlation
            if len(grad_imp) > 1:
                tau, pval = kendalltau(attn, grad_imp)
                if np.isnan(tau):
                    tau = 0.0
                    pval = 1.0
            else:
                tau = 0.0
                pval = 1.0
            
            kendall_taus.append(float(tau))
            kendall_pvalues.append(float(pval))

        except Exception as e:
            print(f"\n  Warning: Sample {idx} failed: {e}")
            kendall_taus.append(0.0)
            kendall_pvalues.append(1.0)

        if (idx + 1) % 500 == 0:
            torch.cuda.empty_cache()

        if (idx + 1) % 2000 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            eta = (len(test_dataset) - idx - 1) / rate
            print(f"\n  [{idx+1}/{len(test_dataset)}] "
                  f"Rate: {rate:.1f} samples/sec | "
                  f"ETA: {eta/60:.1f} min | "
                  f"Avg Tau: {np.mean(kendall_taus):.4f}")

    total_time = time.time() - start_time

    output = {
        'kendall_taus': kendall_taus,
        'kendall_pvalues': kendall_pvalues,
        'total_samples': len(kendall_taus),
        'total_time_seconds': total_time,
        'avg_kendall_tau': float(np.mean(kendall_taus)),
        'std_kendall_tau': float(np.std(kendall_taus)),
        'median_kendall_tau': float(np.median(kendall_taus)),
    }

    output_path = results_dir / "kendall_tau_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
