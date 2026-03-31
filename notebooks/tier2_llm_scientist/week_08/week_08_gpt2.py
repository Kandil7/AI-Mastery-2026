"""
Week 8: GPT-2 Pre-trained Model Loading - Python Script Version

This script demonstrates loading and using pre-trained GPT-2 models.
Convert to Jupyter notebook format using: jupytext --to notebook week_08_gpt2.py

Author: AI-Mastery-2026
"""

# %%
# Imports
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
print("✓ Imports successful")

# %%
# Step 1: Download Pre-trained GPT-2
model_name = 'gpt2'  # 117M parameters (GPT-2 small)

print(f"Loading {model_name}...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Model info
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {model_name}")
print(f"Total parameters: {total_params:,}")
print(f"Model size: {total_params * 4 / 1024**2:.1f} MB (FP32)")

# Architecture details
config = model.config
print(f"\nArchitecture:")
print(f"  Vocabulary size: {config.vocab_size:,}")
print(f"  Max sequence length: {config.n_positions}")
print(f"  Hidden size: {config.n_embd}")
print(f"  Number of layers: {config.n_layer}")
print(f"  Attention heads: {config.n_head}")

print("\n✓ Model loading successful!")

# %%
# Step 2: Understanding Tokenization
text = "Hello, how are you doing today? I'm learning about GPT-2!"

# Tokenize
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"Original text: {text}")
print(f"\nTokens ({len(tokens)}): {tokens}")
print(f"\nToken IDs ({len(token_ids)}): {token_ids}")

# Decode back
decoded = tokenizer.decode(token_ids)
print(f"\nDecoded: {decoded}")

# Set pad token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

print("\n✓ Tokenization explained")

# %%
# Step 3: Text Generation - Greedy Decoding
def generate_greedy(prompt, max_length=50):
    """Generate text using greedy decoding."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Get next token (argmax)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Test
prompt = "The future of artificial intelligence is"
generated = generate_greedy(prompt, max_length=50)

print(f"Prompt: {prompt}")
print(f"\nGreedy Generation:\n{generated}")
print("\n✓ Greedy decoding works (but repetitive!)")

# %%
# Step 4: Advanced Sampling
def generate_with_sampling(prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
    """Generate with top-k and top-p sampling."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_values)
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Compare strategies
prompt = "In a world where AI has surpassed human intelligence,"

print(f"Prompt: {prompt}\n")
print("="*80)

print("\n1. Low Temperature (0.5):")
print(generate_with_sampling(prompt, max_length=80, temperature=0.5, top_k=50, top_p=0.95))

print("\n" + "="*80)
print("\n2. Balanced (temp=1.0):")
print(generate_with_sampling(prompt, max_length=80, temperature=1.0, top_k=50, top_p=0.95))

print("\n✓ Advanced sampling demonstrated")

# %%
print("""
## Week 8 Complete! ✅

### Key Learnings:
1. Loaded pre-trained GPT-2 from Hugging Face
2. Understood BPE tokenization
3. Implemented greedy vs sampling strategies
4. Explored temperature effects

### Interview Takeaways:
- GPT-2 uses causal attention (autoregressive)
- Top-p sampling > greedy for quality
- Temperature controls randomness
- Few-shot learning without fine-tuning

**Next**: Explore GPT-3 API, build chatbots, implement RAG
""")
