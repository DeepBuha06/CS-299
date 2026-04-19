
import torch


class AttentionGradientCapture:

    def __init__(self, model):
        self.attentions = []
        self.attention_grads = []
        self._handles = []
        self._model = model

    def attach(self):
        for layer in self._model.distilbert.transformer.layer:
            sa = layer.attention
            h_fwd = sa.register_forward_hook(self._forward_hook)
            self._handles.append(h_fwd)

    def _forward_hook(self, module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            attn = output[1]  # (batch, num_heads, seq_len, seq_len)
            attn.retain_grad()
            self.attentions.append(attn)

    def collect_grads(self):
        self.attention_grads = []
        for attn in self.attentions:
            if attn.grad is not None:
                self.attention_grads.append(attn.grad.detach())
            else:
                self.attention_grads.append(torch.zeros_like(attn))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.attentions = []
        self.attention_grads = []


def compute_relevance_map(model, input_ids, attention_mask, target_class):
    device = input_ids.device
    model.zero_grad()

    capture = AttentionGradientCapture(model)
    capture.attach()

    outputs = model.distilbert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True
    )
    cls_output = outputs.last_hidden_state[:, 0, :]
    logits = model.classifier(model.dropout(cls_output))

    one_hot = torch.zeros_like(logits)
    one_hot[0, target_class] = 1.0
    logits.backward(gradient=one_hot, retain_graph=False)

    capture.collect_grads()

    attentions = [a.detach() for a in capture.attentions]
    grads = capture.attention_grads

    capture.remove()

    num_layers = len(attentions)
    seq_len = input_ids.shape[1]

    R = torch.eye(seq_len, device=device, dtype=attentions[0].dtype)
    R = R.unsqueeze(0)  # (1, seq_len, seq_len)

    for l in range(num_layers):
        A = attentions[l]     # (1, num_heads, seq_len, seq_len)
        gA = grads[l]         # (1, num_heads, seq_len, seq_len)

        grad_weighted = (gA * A).clamp(min=0)     # Hadamard + ReLU
        A_bar = grad_weighted.mean(dim=1)          # Average across heads: (1, seq, seq)

        row_sum = A_bar.sum(dim=-1, keepdim=True)
        row_sum = row_sum.clamp(min=1e-9)
        A_bar = A_bar / row_sum

        R = R + torch.bmm(A_bar, R)

    relevance = R[0, 0, :]  # (seq_len,)

    relevance = relevance - relevance.min()
    r_max = relevance.max()
    if r_max > 0:
        relevance = relevance / r_max

    return relevance.detach().cpu().numpy()
