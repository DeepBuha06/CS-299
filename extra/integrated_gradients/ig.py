import torch
import numpy as np
from torch.nn import functional as F

def compute_integrated_gradients(model, input_ids, attention_mask, target_class=None, steps=50):
    model.eval()

    # 1. Get embedding layer
    embeddings_layer = model.distilbert.embeddings.word_embeddings

    with torch.no_grad():
        input_embeds = embeddings_layer(input_ids)

        # FIX 1: Use PAD token (id=0) embedding as baseline instead of zeros.
        # zeros are OUT OF DISTRIBUTION for DistilBERT — the model was never trained
        # on zero vectors, so gradients along the zero→input path are noisy/unreliable.
        pad_token_id = torch.zeros_like(input_ids)  # all zeros = PAD token id
        baseline_embeds = embeddings_layer(pad_token_id)

        if target_class is None:
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            target_class = torch.argmax(logits, dim=-1).item()
            # target_class = 0

    # 2. Iterate and interpolate between Baseline (0%) and Actual Input (100%)
    integrated_grads = torch.zeros_like(input_embeds)

    for alpha in np.linspace(0, 1.0, steps):
        interpolated_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated_embeds = interpolated_embeds.detach().requires_grad_(True)

        # (We skip the first embedding layer since we are providing them directly)
        outputs = model.distilbert(inputs_embeds=interpolated_embeds, attention_mask=attention_mask)
        hidden_states = outputs[0]

        cls_output = hidden_states[:, 0, :]
        pooled_output = model.dropout(cls_output)
        logits = model.classifier(pooled_output)

        probs = torch.softmax(logits, dim=-1)
        target_prob = probs[0, target_class]

        # 3. Backward pass — what is the gradient of the interpolated text?
        model.zero_grad()
        target_prob.backward()

        # FIX 2: Explicitly .detach().clone() the gradient before accumulating.
        # references from the previous iteration, causing gradient leakage.
        integrated_grads += interpolated_embeds.grad.detach().clone()

    # 4. Average the gradients across the steps and multiply by the (input - baseline) delta
    avg_grads = integrated_grads / steps
    attributions = (input_embeds - baseline_embeds) * avg_grads

    # 5. Summarize the attribution score for each token by summing across hidden dimensions
    token_attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # embed_delta = (input_embeds - baseline_embeds)                          # how far each word is from PAD
    # embed_norms = embed_delta.norm(dim=-1).squeeze(0).detach().cpu().numpy() # one number per token
    # token_attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    # token_attributions = np.where(embed_norms > 1e-6,
    #                               token_attributions / embed_norms,          # remove the distance bias
    #                               0.0)

    # FIX 3: Normalize while PRESERVING SIGN.
    # (you can no longer tell which words push toward positive vs negative).
    score_sum = np.sum(np.abs(token_attributions))
    if score_sum > 0:
        token_attributions = token_attributions / score_sum

    return token_attributions, target_class


def pool_subword_attributions(tokens, attributions):
    word_tokens = []
    word_scores = []
    current_word = None
    current_score = 0.0

    for token, score in zip(tokens, attributions):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue

        if token.startswith('##'):
            current_score += score
            if current_word is not None:
                current_word += token[2:]  # Reconstruct the full word string
        else:
            if current_word is not None:
                word_tokens.append(current_word)
                word_scores.append(current_score)
            current_word = token
            current_score = score

    if current_word is not None:
        word_tokens.append(current_word)
        word_scores.append(current_score)

    return word_tokens, np.array(word_scores)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from config_transformer import TransformerConfig
    from models_transformer.model import TransformerClassifier
    from models_transformer.dataset import get_tokenizer

    print("Loading model for Integrated Gradients solve...")
    tokenizer = get_tokenizer(TransformerConfig.MODEL_NAME)
    model = TransformerClassifier(model_name=TransformerConfig.MODEL_NAME, num_labels=2)
    checkpoint = torch.load('../../checkpoints/' + TransformerConfig.MODEL_CHECKPOINT, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    text = "The plot was confusing and the acting felt forced, but the cinematography was absolutely stunning and the soundtrack was hauntingly beautiful."
    encoding = tokenizer(text, return_tensors='pt')

    print(f"Running IG on: {text}")
    token_attrs, cls = compute_integrated_gradients(
        model, encoding['input_ids'], encoding['attention_mask']
    )

    raw_tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

    # FIX 4 applied here — pool ##subwords back to whole words before printing
    words, word_attrs = pool_subword_attributions(raw_tokens, token_attrs)

    print(f"\nFinal Class: {'Positive' if cls == 1 else 'Negative'}")
    print("\nIntegrated Gradient Attributions (subword-pooled, sign-preserved):")
    print(f"{'Word':>20}  {'Score':>8}  Direction")
    print("-" * 42)
    for word, score in zip(words, word_attrs):
        direction = "→ POS" if score > 0 else "→ NEG"
        print(f"{word:>20}: {score:>8.4f}  {direction}")

    