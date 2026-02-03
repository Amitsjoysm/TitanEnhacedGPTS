"""Enhanced Titan-GPT v2: GPT Architecture with Advanced Neural Memory

Addresses all critical issues:
1. Separate training and inference modes
2. Clean memory update protocol
3. Gradient-free surprise computation
4. Hierarchical memory integration
5. Improved generation capabilities

Based on:
- 'Titans: Learning to Memorize at Test Time' (arXiv:2501.00663)
- 'Build a Large Language Model (From Scratch)' by Sebastian Raschka
- Modern memory-augmented architectures
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from .neural_memory_v2 import NeuralMemoryV2


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
    """Multi-head self-attention with optional attention weight output."""
    
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
    
    def forward(self, x: torch.Tensor, return_attn_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        attn_weights_dropped = self.dropout(attn_weights)
        
        # Combine heads
        context_vec = (attn_weights_dropped @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        if return_attn_weights:
            return context_vec, attn_weights
        return context_vec, None


class FeedForward(nn.Module):
    """Position-wise feed-forward network with SiLU activation."""
    
    def __init__(self, cfg: Dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.SiLU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TitanTransformerBlockV2(nn.Module):
    """Enhanced transformer block with v2 neural memory.
    
    Improvements:
    - Separate training/inference modes
    - Gradient-free surprise computation
    - Hierarchical memory integration
    - Optional attention weight extraction
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
        
        # Long-term memory (enhanced neural memory v2)
        if use_memory:
            self.neural_memory = NeuralMemoryV2(
                dim=cfg["emb_dim"],
                vocab_size=cfg["vocab_size"],
                short_term_size=cfg.get("short_term_size", 128),
                medium_term_size=cfg.get("medium_term_size", 512),
                long_term_size=cfg.get("long_term_size", 2048),
                num_memory_layers=cfg.get("num_memory_layers", 2),
                surprise_momentum=cfg.get("surprise_momentum", 0.9),
                use_compression=cfg.get("use_compression", True),
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
        token_ids: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        mode: str = "train"
    ) -> torch.Tensor:
        """Forward pass with mode-aware memory operations.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            token_ids: Token indices [batch, seq_len]
            logits: Optional logits for surprise computation
            update_memory: Whether to update memory
            mode: "train" or "inference"
            
        Returns:
            output: Transformed tensor
        """
        # Attention block (short-term memory)
        shortcut = x
        x = self.norm1(x)
        
        # Get attention weights for surprise computation
        attn_out, attn_weights = self.att(x, return_attn_weights=(mode == "train"))
        attn_out = self.drop_shortcut(attn_out)
        
        # Neural memory integration (long-term memory)
        if self.use_memory:
            mem_input = self.norm_mem(shortcut)
            mem_out = self.neural_memory(
                mem_input,
                token_ids=token_ids,
                logits=logits,
                attn_weights=attn_weights,
                update_memory=update_memory,
                mode=mode
            )
            
            # Apply memory variant
            if self.memory_variant == "mac":
                # Memory as context
                attn_out = attn_out + mem_out
            elif self.memory_variant == "mag":
                # Memory as gate
                gate = self.memory_gate(mem_out)
                attn_out = gate * attn_out
            elif self.memory_variant == "hybrid":
                # Both context and gate
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


class TitanGPTModelV2(nn.Module):
    """Enhanced Titan-GPT v2 with all improvements.
    
    Key features:
    - Gradient-free surprise metrics
    - Hierarchical 3-tier memory
    - Separate training/inference modes
    - Memory compression
    - Improved generation
    """
    
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Transformer blocks with enhanced memory
        self.trf_blocks = nn.ModuleList([
            TitanTransformerBlockV2(
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
        update_memory: bool = True,
        mode: str = "train"
    ) -> torch.Tensor:
        """Forward pass with mode specification.
        
        Args:
            in_idx: Input token indices [batch, seq_len]
            targets: Optional target tokens [batch, seq_len]
            update_memory: Whether to update memory
            mode: "train" or "inference"
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = in_idx.shape
        device = in_idx.device
        
        # Embeddings
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # Store input for memory operations
        token_ids = in_idx
        
        # Pass through transformer blocks
        for block in self.trf_blocks:
            # Compute logits before block for surprise metric
            with torch.no_grad():
                pre_logits = self.out_head(self.final_norm(x))
            
            x = block(
                x,
                token_ids=token_ids,
                logits=pre_logits if update_memory else None,
                update_memory=update_memory,
                mode=mode
            )
        
        # Output
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits
    
    def reset_memory(self, level: str = "all"):
        """Reset memory in all blocks.
        
        Args:
            level: "short", "medium", "long", or "all"
        """
        for block in self.trf_blocks:
            if block.use_memory:
                block.neural_memory.reset_memory(level=level)
    
    def set_mode(self, mode: str):
        """Set model mode.
        
        Args:
            mode: "train" or "inference"
        """
        if mode == "train":
            self.train()
        else:
            self.eval()
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_memory: bool = True
    ) -> torch.Tensor:
        """Generate text with enhanced memory.
        
        Args:
            idx: Starting token indices [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k sampling
            use_memory: Whether to use memory during generation
            
        Returns:
            Generated token indices [batch, seq_len + max_new_tokens]
        """
        self.eval()
        
        # Reset short-term memory for fresh generation
        if use_memory:
            self.reset_memory(level="short")
        
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx[:, -self.cfg["context_length"]:]
            
            # Forward pass in inference mode
            with torch.no_grad():
                logits = self.forward(
                    idx_cond,
                    update_memory=use_memory,
                    mode="inference"
                )
            
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
