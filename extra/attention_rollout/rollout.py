"""
Attention Rollout (Abnar & Zuidema, 2020)

Tracks true information flow through a Transformer by recursively
multiplying attention matrices across all layers, accounting for
residual connections via identity matrix addition.
"""

import torch


def compute_rollout(attentions_list, head_fusion='mean', discard_ratio=0.0):
    """
    Compute attention rollout across all transformer layers.

    Args:
        attentions_list: List of attention tensors, one per layer.
            Each tensor: (batch, num_heads, seq_len, seq_len)
        head_fusion: 'mean', 'max', or 'min' — how to combine heads.
        discard_ratio: Fraction of lowest-attention values to zero out
            per layer before rollout (helps sharpen the signal).

    Returns:
        rollout: (batch, seq_len, seq_len) — rolled-out attention matrix.
    """
    num_layers = len(attentions_list)
    batch_size, _, seq_len, _ = attentions_list[0].shape

    rollout = None

    for l in range(num_layers):
        # Fuse across heads
        if head_fusion == 'mean':
            A = attentions_list[l].mean(dim=1)
        elif head_fusion == 'max':
            A = attentions_list[l].max(dim=1).values
        elif head_fusion == 'min':
            A = attentions_list[l].min(dim=1).values
        else:
            A = attentions_list[l].mean(dim=1)

        # Optional: discard lowest attention values to sharpen signal
        if discard_ratio > 0:
            flat = A.view(batch_size, -1)
            k = int(flat.shape[-1] * discard_ratio)
            if k > 0:
                threshold = flat.kthvalue(k, dim=-1).values.unsqueeze(-1)
                flat[flat < threshold] = 0
                A = flat.view(batch_size, seq_len, seq_len)

        # A_res = A + I  (then re-normalize rows)
        I = torch.eye(seq_len, device=A.device, dtype=A.dtype)
        I = I.unsqueeze(0).expand(batch_size, -1, -1)
        A_res = A + I
        A_res = A_res / A_res.sum(dim=-1, keepdim=True)

        if rollout is None:
            rollout = A_res
        else:
            rollout = torch.bmm(A_res, rollout)

    return rollout


def get_cls_rollout(attentions_list, head_fusion='mean', discard_ratio=0.1):
    """
    Get the rollout relevance scores from [CLS] to all input tokens.

    Args:
        attentions_list: List of attention tensors from all layers.
        head_fusion: How to combine heads ('mean', 'max', 'min').
        discard_ratio: Fraction of lowest attention to zero out.

    Returns:
        cls_relevance: (batch, seq_len) — relevance of each token to [CLS].
    """
    rollout = compute_rollout(attentions_list, head_fusion=head_fusion,
                              discard_ratio=discard_ratio)
    cls_relevance = rollout[:, 0, :]
    return cls_relevance
