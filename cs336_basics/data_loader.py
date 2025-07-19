import numpy as np
import torch
from typing import Tuple

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a batch of input sequences and their next-token targets from a 1D numpy array of token IDs.

    Args:
        x (np.ndarray): 1D array of token IDs.
        batch_size (int): Number of sequences to sample.
        context_length (int): Length of each input sequence.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - input tensor of shape (batch_size, context_length)
            - target tensor of shape (batch_size, context_length)
    """
    assert x.ndim == 1, "Input x must be a 1D numpy array."
    max_start = x.shape[0] - context_length - 1
    assert max_start >= 0, "Input array is too short for the given context_length."
    
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    input_seqs = np.stack([x[s:s+context_length] for s in starts])
    target_seqs = np.stack([x[s+1:s+context_length+1] for s in starts])
    input_tensor = torch.tensor(input_seqs, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_seqs, dtype=torch.long, device=device)
    return input_tensor, target_tensor 