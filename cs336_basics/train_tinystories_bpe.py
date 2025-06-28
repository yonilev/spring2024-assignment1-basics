#!/usr/bin/env python3
"""
Train a byte-level BPE tokenizer on the TinyStories dataset.
"""

import json
import pathlib
from cs336_basics.tokenizer import train_bpe
from tests.common import gpt2_bytes_to_unicode


def main():
    # Path to the TinyStories dataset
    input_path = pathlib.Path("tests/fixtures/tinystories_sample_5M.txt")
    
    # Training parameters
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    # Train the BPE tokenizer
    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    
    print(f"Training completed!")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    # Serialize vocabulary to JSON
    vocab_output_path = "tinystories_vocab.json"
    vocab_serializable = {}
    for token_id, token_bytes in vocab.items():
        # Convert bytes to a readable format for JSON serialization
        # We'll use the GPT-2 byte-to-unicode mapping for readability
        byte_to_unicode = gpt2_bytes_to_unicode()
        unicode_repr = ''.join([byte_to_unicode[b] for b in token_bytes])
        vocab_serializable[token_id] = unicode_repr
    
    with open(vocab_output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"Vocabulary saved to: {vocab_output_path}")
    
    # Serialize merges to text file
    merges_output_path = "tinystories_merges.txt"
    with open(merges_output_path, 'w', encoding='utf-8') as f:
        for merge_token_1, merge_token_2 in merges:
            # Convert bytes to unicode representation for readability
            byte_to_unicode = gpt2_bytes_to_unicode()
            unicode_1 = ''.join([byte_to_unicode[b] for b in merge_token_1])
            unicode_2 = ''.join([byte_to_unicode[b] for b in merge_token_2])
            f.write(f"{unicode_1} {unicode_2}\n")
    
    print(f"Merges saved to: {merges_output_path}")

if __name__ == "__main__":
    main() 