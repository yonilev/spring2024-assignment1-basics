import argparse
import numpy as np
from cs336_basics.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Convert a text file to a .npy file of token IDs using a BPE tokenizer.")
    parser.add_argument('--text', type=str, required=True, help='Input text file')
    parser.add_argument('--vocab', type=str, required=True, help='Tokenizer vocab JSON file')
    parser.add_argument('--merges', type=str, required=True, help='Tokenizer merges TXT file')
    parser.add_argument('--special_tokens', type=str, nargs='*', default=["<|endoftext|>"], help='Special tokens for tokenizer')
    parser.add_argument('--out', type=str, required=True, help='Output .npy file for token IDs')
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=args.special_tokens)

    # Read text and encode
    with open(args.text, 'r', encoding='utf-8') as f:
        text = f.read()
    token_ids = tokenizer.encode(text)
    print(f"Encoded {len(token_ids)} tokens.")

    # Save as numpy array
    np.save(args.out, np.array(token_ids, dtype=np.int32))
    print(f"Saved token IDs to {args.out}")

if __name__ == "__main__":
    main() 