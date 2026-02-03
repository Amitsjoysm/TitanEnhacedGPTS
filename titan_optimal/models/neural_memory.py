"""Neural Long-Term Memory Module for Titans Architecture

Implements the neural memory module as described in:
'Titans: Learning to Memorize at Test Time' (arXiv:2501.00663)

Key features:
- Key-value memory storage with online linear regression
- Surprise-based selective memory updates
- Adaptive forgetting mechanism
- Efficient retrieval without full recomputation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SurpriseMetric(nn.Module):
    """Computes surprise metric based on gradient of loss function.
    
    Uses two components:
    - Momentary surprise: from current input
    - Past surprise: incorporates recent context via momentum
    """
    
    def __init__(self, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('past_surprise', None)
    
    def forward(self, loss_grad: torch.Tensor) -> torch.Tensor:
        """Compute surprise from loss gradient.
        
        Args:
            loss_grad: Gradient of loss function [batch, seq_len, dim]
            
        Returns:
            surprise: Surprise metric [batch, seq_len]
        """
        # Momentary surprise: magnitude of gradient
        momentary_surprise = torch.norm(loss_grad, dim=-1)
        
        # Past surprise with momentum
        if self.past_surprise is None:
            self.past_surprise = momentary_surprise
        else:
            self.past_surprise = (
                self.momentum * self.past_surprise + 
                (1 - self.momentum) * momentary_surprise
            )
        
        # Combined surprise
        surprise = momentary_surprise + self.past_surprise
        return surprise


class AdaptiveForgetting(nn.Module):
    """Adaptive weight decay for forgetting less important memories."""
    
    def __init__(self, base_decay: float = 0.01, learnable: bool = True):
        super().__init__()
        if learnable:
            self.decay_param = nn.Parameter(torch.tensor(base_decay))
        else:
            self.register_buffer('decay_param', torch.tensor(base_decay))
    
    def forward(self, memory_importance: torch.Tensor) -> torch.Tensor:
        """Compute forgetting gates based on memory importance.
        
        Args:
            memory_importance: Importance scores [batch, memory_size]
            
        Returns:
            forget_gate: Gates for memory retention [batch, memory_size]
        """
        # Higher importance = less forgetting
        forget_gate = torch.sigmoid(-self.decay_param * memory_importance)
        return forget_gate


class NeuralMemory(nn.Module):
    """Neural Long-Term Memory Module.
    
    Implements key-value memory storage with:
    - Surprise-based selective updates
    - Adaptive forgetting
    - Efficient retrieval via MLP
    """
    
    def __init__(
        self,
        dim: int,
        memory_size: int = 1024,
        num_memory_layers: int = 2,
        surprise_momentum: float = 0.9,
        forget_decay: float = 0.01,
        chunk_size: Optional[int] = None,
        use_1d_conv: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.memory_size = memory_size
        self.chunk_size = chunk_size
        
        # Projections for key, value, query
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(dim, dim)
        
        # Optional 1D depthwise-separable convolution
        if use_1d_conv:
            self.key_conv = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
                nn.Conv1d(dim, dim, kernel_size=1)
            )
            self.value_conv = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
                nn.Conv1d(dim, dim, kernel_size=1)
            )
        else:
            self.key_conv = None
            self.value_conv = None
        
        # MLP for memory retrieval (multi-layer for learning complex patterns)
        memory_mlp_layers = []
        for _ in range(num_memory_layers):
            memory_mlp_layers.extend([
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.LayerNorm(dim)
            ])
        self.memory_mlp = nn.Sequential(*memory_mlp_layers)
        
        # Surprise and forgetting mechanisms
        self.surprise_metric = SurpriseMetric(momentum=surprise_momentum)
        self.adaptive_forget = AdaptiveForgetting(base_decay=forget_decay)
        
        # Memory buffers (initialized during forward pass)
        self.register_buffer('memory_keys', None)
        self.register_buffer('memory_values', None)
        self.register_buffer('memory_importance', None)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
    
    def _init_memory(self, batch_size: int, device: torch.device):
        """Initialize memory buffers."""
        self.memory_keys = torch.zeros(
            batch_size, self.memory_size, self.dim, device=device
        )
        self.memory_values = torch.zeros(
            batch_size, self.memory_size, self.dim, device=device
        )
        self.memory_importance = torch.zeros(
            batch_size, self.memory_size, device=device
        )
    
    def update_memory(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        surprise: torch.Tensor
    ):
        """Update memory with new key-value pairs based on surprise.
        
        Args:
            keys: New keys [batch, seq_len, dim]
            values: New values [batch, seq_len, dim]
            surprise: Surprise scores [batch, seq_len]
        """
        batch_size, seq_len, dim = keys.shape
        
        if self.memory_keys is None:
            self._init_memory(batch_size, keys.device)
        
        # Apply forgetting to existing memory
        forget_gate = self.adaptive_forget(self.memory_importance)
        self.memory_importance = self.memory_importance * forget_gate
        
        # Select most surprising tokens to add to memory
        # Use top-k or threshold
        k = min(seq_len, self.memory_size // 4)  # Update 25% of memory per forward
        top_surprise_values, top_surprise_indices = torch.topk(
            surprise, k=k, dim=-1
        )
        
        # Gather corresponding keys and values
        batch_indices = torch.arange(batch_size, device=keys.device).unsqueeze(1).expand(-1, k)
        new_keys = keys[batch_indices, top_surprise_indices]
        new_values = values[batch_indices, top_surprise_indices]
        
        # Find slots in memory to update (replace least important)
        _, replace_indices = torch.topk(
            self.memory_importance, k=k, dim=-1, largest=False
        )
        
        # Update memory slots
        batch_indices_exp = batch_indices.unsqueeze(-1).expand(-1, -1, dim)
        replace_indices_exp = replace_indices.unsqueeze(-1).expand(-1, -1, dim)
        
        self.memory_keys.scatter_(1, replace_indices_exp, new_keys)
        self.memory_values.scatter_(1, replace_indices_exp, new_values)
        self.memory_importance.scatter_(1, replace_indices, top_surprise_values)
    
    def retrieve_memory(self, queries: torch.Tensor) -> torch.Tensor:
        """Retrieve relevant memories using queries.
        
        Args:
            queries: Query vectors [batch, seq_len, dim]
            
        Returns:
            retrieved: Retrieved memory context [batch, seq_len, dim]
        """
        if self.memory_keys is None:
            # No memory yet, return zeros
            return torch.zeros_like(queries)
        
        # Compute attention scores between queries and memory keys
        # [batch, seq_len, dim] @ [batch, dim, memory_size] -> [batch, seq_len, memory_size]
        scores = torch.bmm(queries, self.memory_keys.transpose(1, 2))
        scores = scores / (self.dim ** 0.5)  # Scaled dot-product
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Retrieve values: [batch, seq_len, memory_size] @ [batch, memory_size, dim]
        retrieved = torch.bmm(attn_weights, self.memory_values)
        
        # Pass through MLP for complex pattern learning
        retrieved = self.memory_mlp(retrieved)
        
        return retrieved
    
    def forward(
        self,
        x: torch.Tensor,
        loss_grad: Optional[torch.Tensor] = None,
        update_memory: bool = True
    ) -> torch.Tensor:
        """Forward pass with memory retrieval and optional update.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            loss_grad: Optional gradient for surprise computation
            update_memory: Whether to update memory (True during training)
            
        Returns:
            output: Input augmented with retrieved memory [batch, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Project to keys, values, queries
        keys = self.key_proj(x)
        values = self.value_proj(x)
        queries = self.query_proj(x)
        
        # Apply optional 1D convolution
        if self.key_conv is not None:
            keys = keys.transpose(1, 2)  # [batch, dim, seq_len]
            keys = self.key_conv(keys)
            keys = keys.transpose(1, 2)  # [batch, seq_len, dim]
            
            values = values.transpose(1, 2)
            values = self.value_conv(values)
            values = values.transpose(1, 2)
        
        # Retrieve from memory
        retrieved = self.retrieve_memory(queries)
        
        # Update memory if training and gradient available
        if update_memory and loss_grad is not None:
            surprise = self.surprise_metric(loss_grad)
            self.update_memory(keys, values, surprise)
        
        # Combine input with retrieved memory (residual connection)
        output = x + self.out_proj(retrieved)
        
        return output
    
    def reset_memory(self):
        """Reset memory buffers (useful between episodes/documents)."""
        self.memory_keys = None
        self.memory_values = None
        self.memory_importance = None
        self.surprise_metric.past_surprise = None
