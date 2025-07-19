import torch
import torch.nn.functional as F
from typing import List, Optional


def top_p_sampling(logits: torch.Tensor, p: float) -> int:
    """
    Perform top-p (nucleus) sampling on logits and return the sampled token index.
    Args:
        logits: 1D tensor of logits (vocab_size,)
        p: cumulative probability threshold (0 < p <= 1)
    Returns:
        int: sampled token index
    """
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative_probs > p
    if torch.any(cutoff):
        cutoff_idx = torch.where(cutoff)[0][0] + 1
        sorted_probs = sorted_probs[:cutoff_idx]
        sorted_indices = sorted_indices[:cutoff_idx]
    sorted_probs = sorted_probs / sorted_probs.sum()  # renormalize
    sampled_idx = torch.multinomial(sorted_probs, 1).item()
    return sorted_indices[sampled_idx].item()


def decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: Optional[str] = None,
    endoftext_token: str = "<|endoftext|>",
) -> str:
    """
    Generate a completion from a language model given a prompt.
    Args:
        model: a TransformerLM instance
        tokenizer: a Tokenizer instance
        prompt: input string to condition on
        max_new_tokens: maximum number of tokens to generate
        temperature: softmax temperature
        top_p: nucleus sampling threshold
        device: torch device (if None, use model's device)
        endoftext_token: string for end-of-text special token
    Returns:
        str: generated completion (including prompt)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)
    # Get endoftext token id
    if hasattr(tokenizer, 'special_tokens') and endoftext_token in tokenizer.special_tokens:
        endoftext_id = tokenizer.encode(endoftext_token)[0]
    else:
        raise ValueError(f"End-of-text token '{endoftext_token}' not found in tokenizer.special_tokens.")
    generated = input_ids.tolist()[0]
    for _ in range(max_new_tokens):
        # Truncate context if needed
        if input_ids.shape[1] > getattr(model, 'context_length', 128):
            input_ids = input_ids[:, -model.context_length:]
        with torch.no_grad():
            logits = model(input_ids)  # (1, seq_len, vocab_size)
        next_token_logits = logits[0, -1, :]
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        # Top-p sampling
        next_token_id = top_p_sampling(next_token_logits, top_p)
        generated.append(next_token_id)
        if next_token_id == endoftext_id:
            break
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
    return tokenizer.decode(generated) 