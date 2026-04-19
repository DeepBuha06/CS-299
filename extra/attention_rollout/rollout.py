import torch

def compute_rollout(attentions_list, head_fusion='mean', discard_ratio=0.0):
    num_layers = len(attentions_list)
    batch_size, _, seq_len, _ = attentions_list[0].shape

    rollout = None

    for l in range(num_layers):
        if head_fusion == 'mean':
            A = attentions_list[l].mean(dim=1)
        elif head_fusion == 'max':
            A = attentions_list[l].max(dim=1).values
        elif head_fusion == 'min':
            A = attentions_list[l].min(dim=1).values
        else:
            A = attentions_list[l].mean(dim=1)

        if discard_ratio > 0:
            flat = A.view(batch_size, -1)
            k = int(flat.shape[-1] * discard_ratio)
            if k > 0:
                threshold = flat.kthvalue(k, dim=-1).values.unsqueeze(-1)
                flat[flat < threshold] = 0
                A = flat.view(batch_size, seq_len, seq_len)

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
    rollout = compute_rollout(attentions_list, head_fusion=head_fusion,
                              discard_ratio=discard_ratio)
    cls_relevance = rollout[:, 0, :]
    return cls_relevance
