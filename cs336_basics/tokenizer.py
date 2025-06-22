#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPE (Byte Pair Encoding) tokenizer implementation.
"""

import regex as re
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the given input text file.
    
    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: A non-negative integer that defines the maximum final vocabulary size
                   (including the initial byte vocabulary, vocabulary items produced from merging,
                   and any special tokens).
        special_tokens: A list of strings to add to the vocabulary. These special tokens do not
                       otherwise affect BPE training.
    
    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes] - The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                  to bytes (token bytes).
            merges: list[tuple[bytes, bytes]] - A list of BPE merges produced from training. Each list
                    item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged
                    with <token2>. The merges should be ordered by order of creation.
    """
    # Read the input file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize vocabulary with byte-level tokens (0-255)
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    
    # Add special tokens to vocabulary
    for special_token in special_tokens:
        token_bytes = special_token.encode('utf-8')
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes
    
    # Initialize merges list
    merges = []
    
    # Use PAT regex to tokenize the text first
    regex = re.compile(PAT)
    initial_tokens_raw = regex.findall(text)
    
    # Count token frequencies
    token_counts = defaultdict(int)
    for token in initial_tokens_raw:
        token_counts[token] += 1
    
    # Convert each unique token to bytes and then to list of single-byte tokens
    token_byte_sequences = {}
    for token in token_counts:
        token_bytes = token.encode('utf-8')
        byte_tokens = [bytes([b]) for b in token_bytes]
        token_byte_sequences[token] = byte_tokens
    
    # BPE training loop
    # Initialize bigram counts and mapping
    bigram_counts = defaultdict(int)
    bigram_to_tokens = defaultdict(set)
    
    # Initial bigram counting
    for token, count in token_counts.items():
        token_bytes = token_byte_sequences[token]
        for bigram in zip(token_bytes[:-1], token_bytes[1:]):
            bigram_counts[bigram] += count
            bigram_to_tokens[bigram].add(token)
    
    # Calculate how many merges we need to perform
    merges_needed = vocab_size - len(vocab)
    
    # Create progress bar
    with tqdm(total=merges_needed, desc="Training BPE", unit="merge") as pbar:
        while len(vocab) < vocab_size:
            if not bigram_counts:
                break
            
            # Find the most frequent bigram, breaking ties lexicographically (prefer greater)
            most_frequent_bigram = max(bigram_counts.items(), key=lambda x: (x[1], x[0]))[0]
            
            # Create new token by concatenating the two tokens
            new_token = most_frequent_bigram[0] + most_frequent_bigram[1]
            
            # Add to vocabulary
            vocab[len(vocab)] = new_token
            
            # Add to merges list
            merges.append(most_frequent_bigram)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"vocab_size": len(vocab)})
            
            # Only update tokens that contain the most frequent bigram
            tokens_to_update = bigram_to_tokens[most_frequent_bigram]
            
            # Update bigram counts incrementally for affected tokens
            for token in tokens_to_update:
                old_token_bytes = token_byte_sequences[token]
                count = token_counts[token]
                
                # Remove ALL old bigrams from counts (since token structure changes)
                for bigram in zip(old_token_bytes[:-1], old_token_bytes[1:]):
                    bigram_counts[bigram] -= count            
                # Apply merge to token
                new_token_bytes = []
                i = 0
                while i < len(old_token_bytes):
                    if i < len(old_token_bytes) - 1 and old_token_bytes[i] == most_frequent_bigram[0] and old_token_bytes[i + 1] == most_frequent_bigram[1]:
                        new_token_bytes.append(new_token)
                        i += 2
                    else:
                        new_token_bytes.append(old_token_bytes[i])
                        i += 1
                
                token_byte_sequences[token] = new_token_bytes
                
                # Add ALL new bigrams to counts (since token structure changed)
                for bigram in zip(new_token_bytes[:-1], new_token_bytes[1:]):
                    bigram_counts[bigram] += count
                    bigram_to_tokens[bigram].add(token)
    
    return vocab, merges