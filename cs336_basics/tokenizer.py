#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPE (Byte Pair Encoding) tokenizer implementation.
"""

import json
import regex as re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Iterable, Iterator
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


class Tokenizer:
    """
    BPE (Byte Pair Encoding) tokenizer implementation.
    
    This tokenizer can encode text into integer IDs and decode integer IDs back to text.
    It supports user-provided special tokens that are never split.
    """
    
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        
        Args:
            vocab: dict[int, bytes] - The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                  to bytes (token bytes).
            merges: list[tuple[bytes, bytes]] - A list of BPE merges. Each list item is a tuple of bytes
                    (<token1>, <token2>), representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
            special_tokens: list[str] | None = None - A list of string special tokens for the tokenizer.
                           These strings will never be split into multiple tokens, and will always be
                           kept as a single token.
        """
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.special_tokens = special_tokens or []

        # Add special tokens to vocabulary if they aren't already there
        for special_token in self.special_tokens:
            token_bytes = special_token.encode('utf-8')
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes

        # Create reverse mapping from bytes to token ID
        self.bytes_to_id = {bytes_val: token_id for token_id, bytes_val in self.vocab.items()}
        
        # Create merge index lookup for O(1) access
        self.merge_to_index = {merge: idx for idx, merge in enumerate(self.merges)}
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges.
        
        Args:
            vocab_filepath: str - Path to the vocabulary JSON file.
            merges_filepath: str - Path to the merges text file.
            special_tokens: list[str] | None = None - A list of string special tokens for the tokenizer.
                           These strings will never be split into multiple tokens, and will always be
                           kept as a single token.
        
        Returns:
            Tokenizer - A tokenizer instance constructed from the provided files.
        """
        # Load vocabulary from JSON file
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Convert string keys to integers and string values to bytes
        vocab = {int(i): t.encode('utf-8') for i, t in vocab_data.items()}
        
        # Load merges from text file
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                t1,t2 = line.strip().split()
                merges.append((t1.encode('utf-8'), t2.encode('utf-8')))
        
        return cls(vocab, merges, special_tokens)
    
    def _split_with_delimiters(self, text: str, delimiters: list[str]) -> list[str]:
        if not delimiters:
            return [text]
        # Sort by length to handle overlapping tokens
        pattern = "(" + "|".join(re.escape(tok) for tok in sorted(delimiters, key=len, reverse=True)) + ")"
        return [seg for seg in re.split(pattern, text) if seg != ""]

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of integer token IDs.
        
        Args:
            text: str - The text to encode.
        
        Returns:
            list[int] - A list of integer token IDs representing the encoded text.
        """
        if not text:
            return []
        encoded_ids = []
        segments = self._split_with_delimiters(text, self.special_tokens)
        for segment in segments:
            if segment in self.special_tokens:
                token_bytes = segment.encode('utf-8')
                encoded_ids.append(self.bytes_to_id[token_bytes])
            else:
                initial_tokens = re.findall(PAT, segment)
                for token in initial_tokens:
                    token_bytes = token.encode('utf-8')
                    byte_tokens = [bytes([b]) for b in token_bytes]
                    # Apply BPE merges
                    while len(byte_tokens) > 1:
                        best_merge = None
                        best_merge_idx = -1
                        best_merge_pos = -1
                        for i in range(len(byte_tokens) - 1):
                            bigram = (byte_tokens[i], byte_tokens[i + 1])
                            if bigram in self.merge_to_index:
                                merge_idx = self.merge_to_index[bigram]
                                if best_merge is None or merge_idx < best_merge_idx:
                                    best_merge = bigram
                                    best_merge_idx = merge_idx
                                    best_merge_pos = i
                        if best_merge is None:
                            break
                        merged_token = best_merge[0] + best_merge[1]
                        byte_tokens[best_merge_pos:best_merge_pos+2] = [merged_token]
                    for byte_token in byte_tokens:
                        encoded_ids.append(self.bytes_to_id[byte_token])
        return encoded_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of integer token IDs back to text.
        
        Args:
            token_ids: list[int] - A list of integer token IDs to decode.
        
        Returns:
            str - The decoded text.
        """
        if not token_ids:
            return ""
        
        # Convert token IDs to bytes
        byte_sequence = [self.vocab[token_id] for token_id in token_ids]
        
        # Concatenate all bytes
        all_bytes = b''.join(byte_sequence)
        
        # Decode to string
        try:
            return all_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback: replace invalid bytes with replacement character
            return all_bytes.decode('utf-8', errors='replace')

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator 
        that lazily yields token IDs. This is required for memory-efficient tokenization 
        of large files that we cannot directly load into memory.
        
        Args:
            iterable: Iterable[str] - An iterable of strings to tokenize.
        
        Yields:
            int - Token IDs one at a time.
        """
        for text in iterable:
            # Encode each piece of text and yield all token IDs
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id