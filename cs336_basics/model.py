import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function as a PyTorch module.
    
    Implementation based on: Dan Hendrycks and Kevin Gimpel. "Bridging nonlinearities and 
    stochastic regularizers with gaussian error linear units", 2016.
    
    GELU is defined as: x * Φ(x), where Φ(x) is the cumulative distribution function
    of the standard normal distribution.
    
    The paper approximates this using: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    where erf is the error function.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GELU activation.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: GELU activated tensor with the same shape as input.
        """
        return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm is a variant of layer normalization that normalizes using the RMS of the activations
    instead of the mean and variance. This makes it more efficient and stable.
    
    Args:
        hidden_size (int): The size of the hidden dimension to normalize.
        eps (float): A small value added to the denominator for numerical stability.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Learnable weight parameter
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size) 
                             or (batch_size, hidden_size).
        
        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        # Calculate RMS (Root Mean Square) along the last dimension
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return x * rms * self.weight


class FFN(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).
    
    The FFN consists of two linear transformations with a GELU activation in between:
    FFN(X) = GELU(XW1)W2
    
    Args:
        hidden_size (int): The size of the hidden dimension.
        intermediate_size (int): The size of the intermediate dimension (typically 4x hidden_size).
    """
    
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()        
        # First linear transformation: hidden_size -> intermediate_size
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        
        # Second linear transformation: intermediate_size -> hidden_size
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)
        
        # GELU activation function
        self.gelu = GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the position-wise feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
        
        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        # Apply first linear transformation and GELU activation
        h = self.gelu(self.w1(x))
        
        # Apply second linear transformation
        output = self.w2(h)
        
        return output

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Numerically stable softmax function.
    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which softmax will be computed.
    Returns:
        torch.Tensor: Softmax probabilities with the same shape as input.
    """
    # Subtract max for numerical stability
    x_max, _ = x.max(dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def scaled_dot_product_attention(
    K: torch.Tensor,     # K: (batch_size, ..., seq_len, d_k)
    Q: torch.Tensor,     # Q: (batch_size, ..., seq_len, d_k)
    V: torch.Tensor,     # V: (batch_size, ..., seq_len, d_v)
    mask: Optional[torch.Tensor] = None, # mask: (seq_len, seq_len)
    pdrop: Optional[float] = None,
) -> torch.Tensor:
    d_k = K.size(-1)  # key dimension (d_k)
    
    # scores: (batch_size, ..., seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    
    # attention_weights: (batch_size, ..., seq_len, seq_len)
    attention_weights = softmax(scores, dim=-1)
    
    # Apply dropout if specified
    if pdrop is not None:
        # Dropout is applied to the attention weights
        attention_weights = torch.dropout(attention_weights, pdrop, train=True)
    
    # output: (batch_size, ..., seq_len, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention module.
    
    Implements the multi-head self-attention mechanism with causal masking as described
    in Vaswani et al. [2017]. The causal mask ensures that each position can only attend
    to previous positions and itself.
    
    Args:
        d_model (int): Dimensionality of the Transformer block inputs.
        num_heads (int): Number of heads to use in multi-head self-attention.
        attn_pdrop (float | None): Dropout rate for softmax-normalized attention probabilities.
    """
    
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: Optional[float] = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of causal multi-head self-attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(x.device)
        attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask, pdrop=self.attn_pdrop)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.output_proj(attn_output)
        return output
    

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.
    
    Implements a Transformer block with pre-norm architecture as described in
    "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020).
    
    The block consists of:
    1. Pre-norm multi-head self-attention with residual connection
    2. Pre-norm feed-forward network with residual connection
    
    Args:
        d_model (int): Dimensionality of the Transformer block inputs.
        num_heads (int): Number of heads to use in multi-head self-attention.
        d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
        attn_pdrop (float | None): Dropout rate for softmax-normalized attention probabilities.
        residual_pdrop (float | None): Dropout rate for embeddings and Transformer block sublayer outputs.
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        attn_pdrop: Optional[float] = None, 
        residual_pdrop: Optional[float] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        
        # Pre-norm layers
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        
        # Multi-head self-attention
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, attn_pdrop)
        
        # Feed-forward network
        self.ffn = FFN(d_model, d_ff)
        
        # Dropout layers for residual connections
        self.residual_dropout = nn.Dropout(residual_pdrop) if residual_pdrop is not None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the pre-norm Transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        # Pre-norm attention with residual connection
        h1 = self.ln1(x)
        h1 = self.attn(h1)
        if self.residual_dropout is not None:
            h1 = self.residual_dropout(h1)
        h1 = x + h1

        # Pre-norm feed-forward with residual connection
        h2 = self.ln2(h1)
        h2 = self.ffn(h2)
        if self.residual_dropout is not None:
            h2 = self.residual_dropout(h2)
        h2 = h1 + h2

        return h2
    

class TransformerLM(nn.Module):
    """
    Transformer Language Model.
    
    Implements a complete Transformer language model as described in §3.1 and illustrated in Figure 1.
    The model consists of:
    1. Token embeddings
    2. Position embeddings
    3. Multiple Transformer blocks
    4. Final layer normalization
    5. Language model head
    
    Args:
        vocab_size (int): The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
        context_length (int): The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer blocks to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop (float | None): Drop-out the attention probabilities (the softmax-normalized attention scores) with this rate.
        residual_pdrop (float | None): Apply dropout to the sum of the token and position embeddings as well as the output of each sub-layer.
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(context_length, d_model)
        
        # Dropout for embeddings
        self.embed_dropout = nn.Dropout(residual_pdrop) if residual_pdrop is not None else None
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                attn_pdrop=attn_pdrop,
                residual_pdrop=residual_pdrop
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.ln_final = RMSNorm(d_model)
        
        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer language model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token indices.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size) with unnormalized logits.
        """
        batch_size, seq_len = x.shape
        
        # Get token embeddings
        token_emb = self.token_embeddings(x)  # (batch_size, seq_len, d_model)
        
        # Get position embeddings
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.position_embeddings(positions)  # (seq_len, d_model)
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, d_model)
        
        # Combine token and position embeddings
        h = token_emb + pos_emb
        
        # Apply dropout to embeddings if specified
        if self.embed_dropout is not None:
            h = self.embed_dropout(h)
        
        # Pass through Transformer blocks
        for layer in self.layers:
            h = layer(h)
        
        # Final layer normalization
        h = self.ln_final(h)
        
        # Language model head
        logits = self.lm_head(h)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
