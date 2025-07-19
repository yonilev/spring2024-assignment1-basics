import streamlit as st
import torch
from cs336_basics.decode import decode
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM
import json
import os

# Hardcoded paths
VOCAB_FILE = "experiments/tinystories_vocab.json"
MERGES_FILE = "experiments/tinystories_merges.txt"
MODEL_CKPT = "experiments/checkpoint.pt"
SPECIAL_TOKENS = ["<|endoftext|>"]

def load_tokenizer():
    return Tokenizer.from_files(VOCAB_FILE, MERGES_FILE, special_tokens=SPECIAL_TOKENS)

def load_model():
    config_path = MODEL_CKPT + ".config.json"
    if not os.path.exists(config_path):
        st.error(f"Model config file not found: {config_path}")
        return None
    with open(config_path, "r") as f:
        config = json.load(f)
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        attn_pdrop=config["attn_pdrop"],
        residual_pdrop=config["residual_pdrop"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(MODEL_CKPT)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    return model


tokenizer = load_tokenizer()
model = load_model()

st.title("CS336 Language Model Chat Demo")
st.session_state["chat_history"] = []
prompt = st.text_area("Your prompt:", "Once upon a time")
max_new_tokens = st.slider("Max new tokens", 1, 128, 32)
temperature = st.slider("Temperature", 0.1, 2.0, 1.0)
top_p = st.slider("Top-p (nucleus sampling)", 0.01, 1.0, 1.0)
if st.button("Generate"):
    output = decode(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    st.session_state["chat_history"].append((prompt, output))
for i, (user, response) in enumerate(st.session_state["chat_history"]):
    st.markdown(f"**Prompt {i+1}:** {user}")
    st.markdown(f"**Model:** {response}")