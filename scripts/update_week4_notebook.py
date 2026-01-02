"""
Script to enhance Week 4 Transformers notebook with advanced attention mechanisms.
"""

import json
from pathlib import Path


def create_advanced_attention_cells():
    """Create notebook cells for advanced attention mechanisms."""
    
    cells = []
    
    # Advanced Attention Section Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# Section 3: Advanced Attention Mechanisms\n",
            "---\n",
            "\n",
            "Modern LLMs like GPT-4, Llama 2, and Mistral use sophisticated attention variants.\n",
            "This section covers the cutting-edge mechanisms implemented in `src/llm/attention.py`.\n"
        ]
    })
    
    # 3.1 Causal Attention
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.1 Causal (Autoregressive) Attention\n",
            "\n",
            "In GPT-style models, tokens can only attend to **previous** tokens.\n",
            "This is achieved using a **causal mask** (lower triangular matrix).\n",
            "\n",
            "```\n",
            "Causal Mask:\n",
            "[1, 0, 0, 0]    Token 1 sees: only itself\n",
            "[1, 1, 0, 0]    Token 2 sees: tokens 1-2\n",
            "[1, 1, 1, 0]    Token 3 sees: tokens 1-3\n",
            "[1, 1, 1, 1]    Token 4 sees: all tokens\n",
            "```\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from src.llm.attention import CausalSelfAttention\n",
            "\n",
            "# Create causal attention\n",
            "d_model, num_heads, block_size = 64, 4, 32\n",
            "causal_attn = CausalSelfAttention(d_model, num_heads, block_size)\n",
            "\n",
            "# Input sequence\n",
            "batch_size, seq_len = 2, 8\n",
            "x = torch.randn(batch_size, seq_len, d_model)\n",
            "\n",
            "# Forward pass\n",
            "output = causal_attn(x)\n",
            "print(f\"Input shape: {x.shape}\")\n",
            "print(f\"Output shape: {output.shape}\")\n",
            "\n",
            "# Visualize the causal mask\n",
            "causal_mask = torch.tril(torch.ones(seq_len, seq_len))\n",
            "plt.figure(figsize=(6, 5))\n",
            "plt.imshow(causal_mask.numpy(), cmap='Blues')\n",
            "plt.title('Causal Attention Mask')\n",
            "plt.xlabel('Key Position')\n",
            "plt.ylabel('Query Position')\n",
            "plt.colorbar(label='Attention Allowed')\n",
            "for i in range(seq_len):\n",
            "    for j in range(seq_len):\n",
            "        plt.text(j, i, int(causal_mask[i, j].item()), ha='center', va='center')\n",
            "plt.show()"
        ]
    })
    
    # 3.2 Rotary Position Embeddings (RoPE)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.2 Rotary Position Embeddings (RoPE)\n",
            "\n",
            "RoPE encodes position through **rotation in complex space**, enabling:\n",
            "- Relative position awareness\n",
            "- Better extrapolation to longer sequences\n",
            "- No learnable position parameters\n",
            "\n",
            "**Used in:** Llama, Mistral, Falcon, CodeLlama\n",
            "\n",
            "### Mathematical Formulation\n",
            "\n",
            "For a query $q$ at position $m$ and key $k$ at position $n$:\n",
            "\n",
            "$$q_m \\cdot k_n = \\text{Re}[(q e^{im\\theta}) \\cdot (k e^{in\\theta})^*] = \\text{Re}[q \\cdot k^* \\cdot e^{i(m-n)\\theta}]$$\n",
            "\n",
            "The attention score depends on **relative position** $(m-n)$, not absolute positions!\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from src.llm.attention import AttentionWithRoPE\n",
            "\n",
            "# Create RoPE attention\n",
            "d_model, num_heads = 64, 4\n",
            "rope_attn = AttentionWithRoPE(d_model, num_heads, max_len=256)\n",
            "\n",
            "# Test forward pass\n",
            "x = torch.randn(2, 16, d_model)\n",
            "output = rope_attn(x)\n",
            "print(f\"RoPE Attention output shape: {output.shape}\")\n",
            "\n",
            "# Visualize RoPE frequencies\n",
            "dim = d_model // num_heads\n",
            "positions = np.arange(64)\n",
            "freqs = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Frequency spectrum\n",
            "ax = axes[0]\n",
            "ax.plot(np.arange(len(freqs)), freqs, 'b-o')\n",
            "ax.set_xlabel('Dimension pair index')\n",
            "ax.set_ylabel('Frequency')\n",
            "ax.set_title('RoPE Frequency Spectrum')\n",
            "ax.set_yscale('log')\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "# Rotation angles for different positions\n",
            "ax = axes[1]\n",
            "for i, freq in enumerate(freqs[:4]):\n",
            "    angles = positions * freq\n",
            "    ax.plot(positions, np.sin(angles), label=f'Dim {i}: freq={freq:.4f}')\n",
            "ax.set_xlabel('Position')\n",
            "ax.set_ylabel('sin(position × frequency)')\n",
            "ax.set_title('Rotation Patterns by Dimension')\n",
            "ax.legend()\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # 3.3 Grouped Query Attention
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.3 Grouped Query Attention (GQA)\n",
            "\n",
            "GQA reduces memory by sharing Key-Value heads across Query heads.\n",
            "\n",
            "**Used in:** Llama 2 (70B uses 8 KV heads for 64 Q heads = 8x reduction)\n",
            "\n",
            "| Variant | Q Heads | KV Heads | Memory Savings |\n",
            "|---------|---------|----------|----------------|\n",
            "| MHA | 32 | 32 | Baseline |\n",
            "| GQA | 32 | 8 | 4x for KV cache |\n",
            "| MQA | 32 | 1 | 32x for KV cache |\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import matplotlib.pyplot as plt\n",
            "from src.llm.attention import GroupedQueryAttention, MultiHeadAttention\n",
            "\n",
            "# Compare MHA vs GQA memory\n",
            "d_model = 512\n",
            "seq_len = 1024\n",
            "batch_size = 8\n",
            "\n",
            "# Standard Multi-Head Attention\n",
            "mha = MultiHeadAttention(d_model, num_heads=32)\n",
            "mha_params = sum(p.numel() for p in mha.parameters())\n",
            "\n",
            "# Grouped Query Attention (8 KV heads instead of 32)\n",
            "gqa = GroupedQueryAttention(d_model, num_heads=32, num_kv_heads=8)\n",
            "gqa_params = sum(p.numel() for p in gqa.parameters())\n",
            "\n",
            "print(f\"MHA parameters: {mha_params:,}\")\n",
            "print(f\"GQA parameters: {gqa_params:,}\")\n",
            "print(f\"Parameter reduction: {(1 - gqa_params/mha_params)*100:.1f}%\")\n",
            "\n",
            "# KV Cache size comparison\n",
            "kv_cache_mha = 2 * batch_size * seq_len * d_model  # K and V\n",
            "kv_cache_gqa = 2 * batch_size * seq_len * (d_model * 8 // 32)  # Reduced KV\n",
            "\n",
            "# Visualization\n",
            "fig, ax = plt.subplots(figsize=(10, 5))\n",
            "methods = ['MHA\\n(32 KV heads)', 'GQA\\n(8 KV heads)', 'MQA\\n(1 KV head)']\n",
            "kv_sizes = [32, 8, 1]  # Relative KV cache size\n",
            "colors = ['#3498db', '#2ecc71', '#e74c3c']\n",
            "\n",
            "bars = ax.bar(methods, kv_sizes, color=colors)\n",
            "ax.set_ylabel('Relative KV Cache Size')\n",
            "ax.set_title('Key-Value Cache Memory Comparison')\n",
            "\n",
            "for bar, size in zip(bars, kv_sizes):\n",
            "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,\n",
            "            f'{size}x', ha='center', fontsize=12, fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # 3.4 Flash Attention Concept
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.4 Flash Attention (Conceptual)\n",
            "\n",
            "Flash Attention achieves **2-4x speedup** by:\n",
            "1. **Tiling**: Process attention in blocks that fit in SRAM\n",
            "2. **Recomputation**: Recompute during backward pass instead of storing\n",
            "3. **Kernel fusion**: Combine ops to reduce memory transfers\n",
            "\n",
            "```\n",
            "Standard Attention:           Flash Attention:\n",
            "┌─────────────────┐           ┌─────────────────┐\n",
            "│   Full QK^T     │ O(n²)     │   Block 1  │ │ │ O(n²/B)\n",
            "│   in HBM        │ memory   │   in SRAM  │ │ │ per tile\n",
            "└─────────────────┘           └─────────────────┘\n",
            "```\n",
            "\n",
            "**Impact:** Enables training with 16x longer sequences!\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "from src.llm.attention import FlashAttention\n",
            "\n",
            "# Create Flash Attention (simplified implementation)\n",
            "flash_attn = FlashAttention(d_model=128, num_heads=8)\n",
            "\n",
            "# Test with longer sequence\n",
            "x = torch.randn(4, 256, 128)\n",
            "output = flash_attn(x)\n",
            "print(f\"Flash Attention output: {output.shape}\")\n",
            "\n",
            "# Memory comparison visualization\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "\n",
            "seq_lengths = [512, 1024, 2048, 4096, 8192]\n",
            "standard_mem = [s**2 for s in seq_lengths]  # O(n²)\n",
            "flash_mem = [s * 256 for s in seq_lengths]  # O(n × block_size)\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "plt.plot(seq_lengths, [m/1e6 for m in standard_mem], 'b-o', label='Standard Attention', lw=2)\n",
            "plt.plot(seq_lengths, [m/1e6 for m in flash_mem], 'g-s', label='Flash Attention', lw=2)\n",
            "plt.xlabel('Sequence Length')\n",
            "plt.ylabel('Memory (millions of elements)')\n",
            "plt.title('Memory Scaling: Standard vs Flash Attention')\n",
            "plt.legend()\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.yscale('log')\n",
            "plt.show()\n",
            "\n",
            "print(f\"\\nAt seq_len=8192:\")\n",
            "print(f\"  Standard: {standard_mem[-1]/1e6:.1f}M elements\")\n",
            "print(f\"  Flash: {flash_mem[-1]/1e6:.1f}M elements\")\n",
            "print(f\"  Reduction: {standard_mem[-1]/flash_mem[-1]:.0f}x\")"
        ]
    })
    
    # Mini-GPT Section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# Section 4: Building a Mini-GPT\n",
            "---\n",
            "\n",
            "Now let's assemble these components into a working **GPT-style decoder block**.\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "from src.llm.attention import CausalSelfAttention\n",
            "\n",
            "class MiniGPTBlock(nn.Module):\n",
            "    \"\"\"A single GPT decoder block.\"\"\"\n",
            "    \n",
            "    def __init__(self, d_model: int, num_heads: int, block_size: int, dropout: float = 0.1):\n",
            "        super().__init__()\n",
            "        self.ln1 = nn.LayerNorm(d_model)\n",
            "        self.attn = CausalSelfAttention(d_model, num_heads, block_size, dropout)\n",
            "        self.ln2 = nn.LayerNorm(d_model)\n",
            "        self.mlp = nn.Sequential(\n",
            "            nn.Linear(d_model, 4 * d_model),\n",
            "            nn.GELU(),\n",
            "            nn.Linear(4 * d_model, d_model),\n",
            "            nn.Dropout(dropout),\n",
            "        )\n",
            "    \n",
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n",
            "        # Pre-norm architecture (like GPT-2)\n",
            "        x = x + self.attn(self.ln1(x))\n",
            "        x = x + self.mlp(self.ln2(x))\n",
            "        return x\n",
            "\n",
            "\n",
            "class MiniGPT(nn.Module):\n",
            "    \"\"\"Minimal GPT model.\"\"\"\n",
            "    \n",
            "    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 4,\n",
            "                 num_layers: int = 4, block_size: int = 128):\n",
            "        super().__init__()\n",
            "        self.token_emb = nn.Embedding(vocab_size, d_model)\n",
            "        self.pos_emb = nn.Embedding(block_size, d_model)\n",
            "        self.blocks = nn.ModuleList([\n",
            "            MiniGPTBlock(d_model, num_heads, block_size)\n",
            "            for _ in range(num_layers)\n",
            "        ])\n",
            "        self.ln_f = nn.LayerNorm(d_model)\n",
            "        self.head = nn.Linear(d_model, vocab_size, bias=False)\n",
            "        self.block_size = block_size\n",
            "    \n",
            "    def forward(self, idx: torch.Tensor) -> torch.Tensor:\n",
            "        B, T = idx.shape\n",
            "        tok_emb = self.token_emb(idx)  # (B, T, d_model)\n",
            "        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)\n",
            "        x = tok_emb + pos_emb\n",
            "        \n",
            "        for block in self.blocks:\n",
            "            x = block(x)\n",
            "        \n",
            "        x = self.ln_f(x)\n",
            "        logits = self.head(x)  # (B, T, vocab_size)\n",
            "        return logits\n",
            "\n",
            "\n",
            "# Test the model\n",
            "model = MiniGPT(vocab_size=1000, d_model=128, num_heads=4, num_layers=2, block_size=64)\n",
            "total_params = sum(p.numel() for p in model.parameters())\n",
            "\n",
            "# Forward pass\n",
            "idx = torch.randint(0, 1000, (2, 32))  # Batch of 2, seq_len 32\n",
            "logits = model(idx)\n",
            "\n",
            "print(f\"MiniGPT Parameters: {total_params:,}\")\n",
            "print(f\"Input shape: {idx.shape}\")\n",
            "print(f\"Output logits shape: {logits.shape}\")\n",
            "print(f\"\\nArchitecture:\")\n",
            "print(f\"  - Token embedding: 1000 × 128\")\n",
            "print(f\"  - Position embedding: 64 × 128\")\n",
            "print(f\"  - Transformer blocks: 2\")\n",
            "print(f\"  - Each block: CausalSelfAttention + MLP\")"
        ]
    })
    
    # Interview questions
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# Section 5: Interview Questions - Advanced Attention\n",
            "---\n",
            "\n",
            "**Q1: Why does Flash Attention achieve O(1) memory for attention weights?**\n",
            "\n",
            "A: Flash Attention never materializes the full N×N attention matrix. It computes attention in tiles,\n",
            "keeping only one tile in SRAM at a time. The numerator and denominator for softmax are accumulated\n",
            "across tiles using the online softmax trick.\n",
            "\n",
            "---\n",
            "\n",
            "**Q2: How does RoPE enable extrapolation to longer sequences?**\n",
            "\n",
            "A: RoPE encodes position through rotation, so the attention score depends only on **relative position**\n",
            "(m-n), not absolute positions. The model can theoretically handle any position as long as it can\n",
            "represent the relative distance. (Though in practice, NTK-aware scaling or ALiBi may be needed.)\n",
            "\n",
            "---\n",
            "\n",
            "**Q3: What's the trade-off between MHA, GQA, and MQA?**\n",
            "\n",
            "| Method | KV Cache | Quality | Use Case |\n",
            "|--------|----------|---------|----------|\n",
            "| MHA | Largest | Best | Training, smaller models |\n",
            "| GQA | Medium | Good | Llama 2 70B, balance |\n",
            "| MQA | Smallest | Lower | Ultra-fast inference |\n",
            "\n",
            "---\n",
            "\n",
            "**Q4: Why is pre-norm (layernorm before attention) preferred in modern LLMs?**\n",
            "\n",
            "A: Pre-norm creates a more direct gradient path (like ResNet) and improves training stability,\n",
            "especially for deep models. Post-norm (original Transformer) can have gradient issues in very deep networks.\n"
        ]
    })
    
    return cells


def update_notebook():
    """Update the Week 4 notebook with advanced attention sections."""
    notebook_path = Path(r"k:\learning\technical\ai-ml\AI-Mastery-2026\research\week4-transformers\week4_transformers.ipynb")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    new_cells = create_advanced_attention_cells()
    notebook['cells'].extend(new_cells)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Successfully updated {notebook_path}")
    print(f"Added {len(new_cells)} new cells")


if __name__ == "__main__":
    update_notebook()
