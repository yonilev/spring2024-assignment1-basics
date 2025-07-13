import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross entropy loss between logits and targets.
    Args:
        logits: Tensor of shape (..., vocab_size), where ... are batch-like dimensions.
        targets: Tensor of shape (...), containing integer indices of the correct class.
    Returns:
        Scalar tensor: average cross entropy loss over all batch elements.
    """
    # logits_stable: (..., vocab_size)
    logits_stable = logits - logits.max(dim=-1, keepdim=True).values
    # exp_logits: (..., vocab_size)
    exp_logits = torch.exp(logits_stable)
    # sum_exp: (...,)
    sum_exp = exp_logits.sum(dim=-1)
    # logsumexp: (...,)
    logsumexp = torch.log(sum_exp)
    # target_logits: (...,)
    target_logits = logits_stable.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # loss: (...,)
    loss = -target_logits + logsumexp
    # returns scalar
    return loss.mean() 