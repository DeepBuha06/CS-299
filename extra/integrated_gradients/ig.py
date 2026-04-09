import torch
import numpy as np
from torch.nn import functional as F

def compute_integrated_gradients(model, input_ids, attention_mask, target_class=None, steps=50):
    """
    Computes Integrated Gradients for a HuggingFace Transformer.
    This completely bypasses the Attention matrices and looks only at the raw Input Embeddings.
    """
    model.eval()
    
    # 1. Get embedding layer
    embeddings_layer = model.distilbert.embeddings.word_embeddings
    
    with torch.no_grad():
        # Get the actual embeddings for the words
        input_embeds = embeddings_layer(input_ids)
        
        # Create a baseline (zeros, representing empty space/no words)
        baseline_embeds = torch.zeros_like(input_embeds)
        
        # Determine the target class if not provided
        if target_class is None:
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            target_class = torch.argmax(logits, dim=-1).item()

    # 2. Iterate and interpolate between Baseline (0) and Actual Input (100)
    # We will accumulate the gradients across these steps
    integrated_grads = torch.zeros_like(input_embeds)
    
    for alpha in np.linspace(0, 1.0, steps):
        # Create the interpolated embedding (e.g., 10% opacity, 20% opacity...)
        interpolated_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated_embeds.requires_grad_(True)
        
        # Manually pass the embeddings to the model 
        # (We skip the first embedding layer since we are providing them directly)
        outputs = model.distilbert(inputs_embeds=interpolated_embeds, attention_mask=attention_mask)
        hidden_states = outputs[0]
        
        # Pass through the classifier head
        # Use [CLS] token representation (first token)
        cls_output = hidden_states[:, 0, :]
        pooled_output = model.dropout(cls_output)
        logits = model.classifier(pooled_output)
        
        # Get the probability for our target class
        probs = torch.softmax(logits, dim=-1)
        target_prob = probs[0, target_class]
        
        # 3. Backward pass! What is the gradient of the interpolated text?
        model.zero_grad()
        target_prob.backward()
        
        # Accumulate the gradient
        integrated_grads += interpolated_embeds.grad
        
    # 4. Average the gradients across the steps and multiply by the actual input
    avg_grads = integrated_grads / steps
    attributions = (input_embeds - baseline_embeds) * avg_grads
    
    # 5. Summarize the attribution scores for each word by summing across the hidden dimensions
    # We take the absolute value to measure total impact (positive or negative) on the sentiment
    word_attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    
    # Normalize the scores to percentages
    score_sum = np.sum(np.abs(word_attributions))
    if score_sum > 0:
        word_attributions = word_attributions / score_sum
        
    return word_attributions, target_class

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
    attrs, cls = compute_integrated_gradients(model, encoding['input_ids'], encoding['attention_mask'])
    
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    print(f"\nFinal Class: {'Positive' if cls == 1 else 'Negative'}")
    print("\nIntegrated Gradient Attributions (Immune to Sponge Effect):")
    for t, a in zip(tokens, attrs):
        if t not in ['[CLS]', '[SEP]', '[PAD]']:
            print(f"{t:>15}: {a:.4f}")
