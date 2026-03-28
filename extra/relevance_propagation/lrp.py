"""
Transformer Relevance Propagation (Chefer et al., 2021)

Combines attention weights with their gradients to isolate the positive
flow of relevance through a Transformer. Produces class-discriminative
token-level relevance maps.
"""

import torch


class AttentionGradientCapture:
    """
    Hooks to capture attention weights (forward) and their gradients (backward)
    from every transformer layer during a single forward+backward pass.
    """

    def __init__(self, model):
        """
        Args:
            model: TransformerClassifier with model.distilbert.transformer.layer
        """
        self.attentions = []
        self.attention_grads = []
        self._handles = []
        self._model = model

    def attach(self):
        """Register hooks on all transformer layers."""
        for layer in self._model.distilbert.transformer.layer:
            sa = layer.attention
            # Forward hook: capture attention weights
            h_fwd = sa.register_forward_hook(self._forward_hook)
            self._handles.append(h_fwd)

    def _forward_hook(self, module, input, output):
        # DistilBERT attention returns (context, attention_probs)
        if isinstance(output, tuple) and len(output) >= 2:
            attn = output[1]  # (batch, num_heads, seq_len, seq_len)
            attn.retain_grad()
            self.attentions.append(attn)

    def collect_grads(self):
        """Call AFTER backward() to extract gradients from retained tensors."""
        self.attention_grads = []
        for attn in self.attentions:
            if attn.grad is not None:
                self.attention_grads.append(attn.grad.detach())
            else:
                self.attention_grads.append(torch.zeros_like(attn))

    def remove(self):
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []
        self.attentions = []
        self.attention_grads = []


def compute_relevance_map(model, input_ids, attention_mask, target_class):
    """
    Compute per-token relevance using gradient-weighted attention rollout.

    Args:
        model: TransformerClassifier (eval mode is fine, gradients flow through attention)
        input_ids: (1, seq_len) token IDs
        attention_mask: (1, seq_len)
        target_class: int, 0 or 1

    Returns:
        relevance_scores: (seq_len,) numpy array — per-token relevance.
            Higher = more relevant to the target class prediction.
    """
    device = input_ids.device
    model.zero_grad()

    # Attach hooks
    capture = AttentionGradientCapture(model)
    capture.attach()

    # Forward pass WITH attention output
    outputs = model.distilbert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True
    )
    cls_output = outputs.last_hidden_state[:, 0, :]
    logits = model.classifier(model.dropout(cls_output))

    # Backward pass w.r.t. target class
    one_hot = torch.zeros_like(logits)
    one_hot[0, target_class] = 1.0
    logits.backward(gradient=one_hot, retain_graph=False)

    # Collect gradients
    capture.collect_grads()

    # Get raw attentions from the forward hook captures
    attentions = [a.detach() for a in capture.attentions]
    grads = capture.attention_grads

    capture.remove()

    num_layers = len(attentions)
    seq_len = input_ids.shape[1]

    # Initialize relevance as identity matrix: R^(L) = I
    R = torch.eye(seq_len, device=device, dtype=attentions[0].dtype)
    R = R.unsqueeze(0)  # (1, seq_len, seq_len)

    for l in range(num_layers):
        A = attentions[l]     # (1, num_heads, seq_len, seq_len)
        gA = grads[l]         # (1, num_heads, seq_len, seq_len)

        # Ā = E_h[ (∇A ⊙ A)^+ ]
        grad_weighted = (gA * A).clamp(min=0)     # Hadamard + ReLU
        A_bar = grad_weighted.mean(dim=1)          # Average across heads: (1, seq, seq)

        # Re-normalize rows
        row_sum = A_bar.sum(dim=-1, keepdim=True)
        row_sum = row_sum.clamp(min=1e-9)
        A_bar = A_bar / row_sum

        # R^(l-1) = R^(l) + Ā × R^(l)
        R = R + torch.bmm(A_bar, R)

    # Final relevance: row 0 = CLS token relevance over input tokens
    relevance = R[0, 0, :]  # (seq_len,)

    # Normalize to [0, 1]
    relevance = relevance - relevance.min()
    r_max = relevance.max()
    if r_max > 0:
        relevance = relevance / r_max

    return relevance.detach().cpu().numpy()
