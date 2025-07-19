import argparse
import os
import numpy as np
import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.data_loader import get_batch
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.optimizer import AdamW, gradient_clipping, get_lr_cosine_schedule
from cs336_basics.checkpoints import save_checkpoint, load_checkpoint
import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    parser.add_argument('--train_data', type=str, required=True, help='Path to tokenized training data (.npy, 1D array of token IDs)')
    parser.add_argument('--val_data', type=str, required=True, help='Path to tokenized validation data (.npy, 1D array of token IDs)')
    parser.add_argument('--vocab', type=str, required=True, help='Path to tokenizer vocab JSON file')
    parser.add_argument('--merges', type=str, required=True, help='Path to tokenizer merges TXT file')
    parser.add_argument('--special_tokens', type=str, nargs='*', default=["<|endoftext|>"], help='Special tokens for tokenizer')
    parser.add_argument('--context_length', type=int, default=128, help='Context length (sequence length)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Transformer layers')
    parser.add_argument('--d_model', type=int, default=256, help='Model embedding dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feedforward hidden dimension')
    parser.add_argument('--attn_pdrop', type=float, default=0.1, help='Attention dropout rate')
    parser.add_argument('--residual_pdrop', type=float, default=0.1, help='Residual dropout rate')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate after annealing')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.95), help='AdamW betas')
    parser.add_argument('--eps', type=float, default=1e-8, help='AdamW epsilon')
    parser.add_argument('--max_iters', type=int, default=10000, help='Total training iterations')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluate/validate every N iterations')
    parser.add_argument('--log_interval', type=int, default=100, help='Log training loss every N iterations')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to save checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint if available')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Max L2 norm for gradient clipping (0 to disable)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0, help='Unused (for compatibility)')
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases (if installed)')
    parser.add_argument('--wandb_project', type=str, default='cs336-basics', help='wandb project name')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    args = parse_args()
    logging.info(f"Using device: {args.device}")

    # Optionally import wandb
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            logging.warning("wandb not installed, continuing without external logging.")
            args.wandb = False

    # Load tokenizer
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=args.special_tokens)
    vocab_size = len(tokenizer.vocab)
    logging.info(f"Loaded tokenizer with vocab size: {vocab_size}")

    # Load data with np.memmap for memory efficiency
    train_tokens = np.load(args.train_data, mmap_mode='r')
    val_tokens = np.load(args.val_data, mmap_mode='r')
    logging.info(f"Train tokens: {train_tokens.shape}, Val tokens: {val_tokens.shape}")

    # Build model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
    ).to(args.device)

    # Build optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Learning rate schedule parameters
    T_w = min(1000, args.max_iters // 10)  # warmup steps
    T_c = args.max_iters  # cosine annealing until end

    # Optionally resume from checkpoint
    start_iter = 0
    if args.resume and os.path.exists(args.checkpoint_path):
        logging.info(f"Resuming from checkpoint: {args.checkpoint_path}")
        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
        logging.info(f"Resumed at iteration {start_iter}")

    # Training loop
    for it in range(start_iter, args.max_iters):
        model.train()
        # Sample batch
        x, y = get_batch(train_tokens, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        # Update learning rate
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, T_w, T_c)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # Logging
        if (it + 1) % args.log_interval == 0 or it == 0:
            logging.info(f"Iter {it+1}/{args.max_iters} | Train loss: {loss.item():.4f} | LR: {lr:.6f}")
            if args.wandb:
                wandb.log({"train/loss": loss.item(), "lr": lr, "iter": it+1})

        # Validation
        if (it + 1) % args.eval_interval == 0 or (it + 1) == args.max_iters:
            model.eval()
            with torch.no_grad():
                val_x, val_y = get_batch(val_tokens, args.batch_size, args.context_length, args.device)
                val_logits = model(val_x)
                val_loss = cross_entropy_loss(val_logits, val_y)
            logging.info(f"Iter {it+1}/{args.max_iters} | Val loss: {val_loss.item():.4f}")
            if args.wandb:
                wandb.log({"val/loss": val_loss.item(), "iter": it+1})
            model.train()

        # Checkpointing
        if (it + 1) % args.eval_interval == 0 or (it + 1) == args.max_iters:
            save_checkpoint(model, optimizer, it + 1, args.checkpoint_path)
            logging.info(f"Checkpoint saved at iter {it+1}")

    logging.info("Training complete.")

if __name__ == "__main__":
    main()
