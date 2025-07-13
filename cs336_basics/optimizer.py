import torch
from torch.optim import Optimizer
import math
# Explicit import for torch.norm

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)  # First moment vector
                    state['v'] = torch.zeros_like(p.data)  # Second moment vector

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']

                state['step'] += 1
                t = state['step']

                # m ← β1m + (1 − β1)g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v ← β2v + (1 − β2)g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected learning rate
                alpha_t = group['lr'] * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)

                # θ ← θ − αt * m / (sqrt(v) + eps)
                denom = v.sqrt().add(group['eps'])
                step = m / denom
                p.data.add_(step, alpha=-alpha_t)

                # θ ← θ − αλθ (Apply weight decay after Adam update)
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss 


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    """
    Clips the gradients of the given parameters so that their global l2-norm does not exceed max_l2_norm.
    Modifies gradients in-place.

    Args:
        parameters: Iterable of torch.nn.Parameter
        max_l2_norm: float, maximum allowed l2-norm
        eps: float, small value for numerical stability (default 1e-6)
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    # Compute global norm
    total_norm = torch.sqrt(torch.stack([g.detach().float().pow(2).sum() for g in grads]).sum())
    if total_norm < max_l2_norm:
        return
    scale = max_l2_norm / (total_norm + eps)
    for g in grads:
        g.mul_(scale)


def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    """
    Compute the learning rate at iteration t using cosine annealing with warmup.

    Args:
        t (int): Current iteration (0-based).
        alpha_max (float): Maximum learning rate.
        alpha_min (float): Minimum (final) learning rate.
        T_w (int): Number of warm-up iterations.
        T_c (int): Number of cosine annealing iterations (inclusive).

    Returns:
        float: The learning rate at iteration t.
    """
    if t < T_w:
        # Warm-up phase
        return (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        # Cosine annealing phase
        cosine = 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w)))
        return alpha_min + cosine * (alpha_max - alpha_min)
    else:
        # Post-annealing phase
        return alpha_min 
