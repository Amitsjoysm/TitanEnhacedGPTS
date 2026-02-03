"""Titan-GPT: GPT Architecture Enhanced with Neural Long-Term Memory

Combines standard GPT transformer blocks with neural memory modules
to enable extended context processing and long-term memorization.

Based on:
- 'Titans: Learning to Memorize at Test Time' (arXiv:2501.00663)
- 'Build a Large Language Model (From Scratch)' by Sebastian Raschka
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from .neural_memory import NeuralMemory


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale and shift."""
    
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism (short-term memory)."""
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape
        
        # Project to queries, keys, values
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # Split into multiple heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)
        
        # Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        
        # Combine heads
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec


class FeedForward(nn.Module):
    """Position-wise feed-forward network with SiLU activation."""
    
    def __init__(self, cfg: Dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.SiLU(),  # Using SiLU as per Titans paper
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TitanTransformerBlock(nn.Module):
    """Transformer block enhanced with neural memory.
    
    Three variants:
    1. MAC (Memory As Context): Memory as additional context
    2. MAG (Memory As Gate): Memory gates the attention output
    3. Hybrid: Both context and gating
    """
    
    def __init__(
        self,
        cfg: Dict,
        memory_variant: str = "mac",
        use_memory: bool = True
    ):
        super().__init__()
        self.use_memory = use_memory
        self.memory_variant = memory_variant
        
        # Short-term memory (attention)
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        
        # Long-term memory (neural memory)
        if use_memory:
            self.neural_memory = NeuralMemory(
                dim=cfg["emb_dim"],
                memory_size=cfg.get("memory_size", 1024),
                num_memory_layers=cfg.get("num_memory_layers", 2),
                surprise_momentum=cfg.get("surprise_momentum", 0.9),
                forget_decay=cfg.get("forget_decay", 0.01),
                use_1d_conv=cfg.get("use_1d_conv", True)
            )
            
            # For MAG variant: gating mechanism
            if memory_variant in ["mag", "hybrid"]:
                self.memory_gate = nn.Sequential(
                    nn.Linear(cfg["emb_dim"], cfg["emb_dim"]),
                    nn.Sigmoid()
                )
        
        # Feed-forward network
        self.ff = FeedForward(cfg)
        
        # Layer normalization
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        if use_memory:
            self.norm_mem = LayerNorm(cfg["emb_dim"])
        
        # Dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(
        self,
        x: torch.Tensor,
        loss_grad: Optional[torch.Tensor] = None,
        update_memory: bool = True
    ) -> torch.Tensor:
        # Attention block (short-term memory)
        shortcut = x
        x = self.norm1(x)
        attn_out = self.att(x)
        attn_out = self.drop_shortcut(attn_out)
        
        # Neural memory integration (long-term memory)
        if self.use_memory:
            mem_input = self.norm_mem(shortcut)
            mem_out = self.neural_memory(
                mem_input,
                loss_grad=loss_grad,
                update_memory=update_memory
            )
            
            # Apply memory variant
            if self.memory_variant == "mac":
                # Memory as context: add to attention output
                attn_out = attn_out + mem_out
            elif self.memory_variant == "mag":
                # Memory as gate: gate the attention output
                gate = self.memory_gate(mem_out)
                attn_out = gate * attn_out
            elif self.memory_variant == "hybrid":
                # Both: add and gate
                gate = self.memory_gate(mem_out)
                attn_out = gate * attn_out + mem_out
        
        x = shortcut + attn_out
        
        # Feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x


class TitanGPTModel(nn.Module):
    """Complete Titan-GPT model with neural long-term memory.
    
    Extends standard GPT with:
    - Neural long-term memory modules
    - Surprise-based selective memory updates
    - Adaptive forgetting
    - Extended context window support
    """
    
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Transformer blocks with memory
        self.trf_blocks = nn.ModuleList([
            TitanTransformerBlock(
                cfg,
                memory_variant=cfg.get("memory_variant", "mac"),
                use_memory=cfg.get("use_memory", True)
            )
            for _ in range(cfg["n_layers"])
        ])
        
        # Output head
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    
    def forward(
        self,
        in_idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        update_memory: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        
        # Embeddings
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # Compute loss gradient for surprise metric (if targets provided)
        loss_grad = None
        if targets is not None and update_memory:
            # Enable gradient computation for embeddings
            x.requires_grad_(True)
        
        # Pass through transformer blocks
        for block in self.trf_blocks:
            x = block(x, loss_grad=loss_grad, update_memory=update_memory)
        
        # Output
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        # Compute loss gradient if needed
        if targets is not None and update_memory:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.cfg["vocab_size"]),
                targets.view(-1),
                reduction='mean'
            )
            if x.requires_grad:
                loss_grad = torch.autograd.grad(
                    loss, x, retain_graph=True, create_graph=False
                )[0]
        
        return logits
    
    def reset_memory(self):
        """Reset all memory buffers in all blocks."""
        for block in self.trf_blocks:
            if block.use_memory:
                block.neural_memory.reset_memory()
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text autoregressively.
        
        Args:
            idx: Starting token indices [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k sampling
            
        Returns:
            Generated token indices [batch, seq_len + max_new_tokens]
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx[:, -self.cfg["context_length"]:]
            
            # Forward pass (no memory updates during generation)
            with torch.no_grad():
                logits = self.forward(idx_cond, update_memory=False)
            
            # Focus on last time step
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
