"""Modular Memory Components for Titan-GPT V3

This module contains reusable, optimized components for the V3 memory system:
1. CircularTensorBuffer - GPU-efficient buffer with batch support
2. SurpriseDrivenConsolidation - Adaptive consolidation based on surprise
3. EarlyStoppingRetrieval - Confidence-based retrieval optimization
4. CrossAttentionRetrieval - Learned cross-attention for memory access
5. ImportanceDecay - Exponential decay for old memories
6. MemoryDiversityLoss - Auxiliary loss for coverage
7. EpisodicBoundaryDetector - Detect topic/document shifts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class CircularTensorBuffer(nn.Module):
    """GPU-efficient circular buffer with proper batch handling.
    
    Replaces Python deque with tensor-based circular buffer for speed.
    Maintains separate buffers per batch item.
    """
    
    def __init__(self, batch_size: int, buffer_size: int, dim: int):
        super().__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.dim = dim
        
        # Circular buffers (one per batch item)
        self.register_buffer(
            'keys_buffer',
            torch.zeros(batch_size, buffer_size, dim)
        )
        self.register_buffer(
            'values_buffer',
            torch.zeros(batch_size, buffer_size, dim)
        )
        self.register_buffer(
            'write_positions',
            torch.zeros(batch_size, dtype=torch.long)
        )
        self.register_buffer(
            'filled_counts',
            torch.zeros(batch_size, dtype=torch.long)
        )
    
    def add(self, keys: torch.Tensor, values: torch.Tensor):
        """Add new memories to circular buffer.
        
        Args:
            keys: [batch, seq_len, dim]
            values: [batch, seq_len, dim]
        """
        batch_size, seq_len, dim = keys.shape
        device = keys.device
        
        # Ensure buffers are on correct device
        if self.keys_buffer.device != device:
            self.keys_buffer = self.keys_buffer.to(device)
            self.values_buffer = self.values_buffer.to(device)
            self.write_positions = self.write_positions.to(device)
            self.filled_counts = self.filled_counts.to(device)
        
        # Handle each batch item and sequence position
        for b in range(batch_size):
            for s in range(seq_len):
                pos = self.write_positions[b].item()
                
                # Write to circular buffer
                self.keys_buffer[b, pos] = keys[b, s].detach()
                self.values_buffer[b, pos] = values[b, s].detach()
                
                # Update position (circular)
                self.write_positions[b] = (pos + 1) % self.buffer_size
                
                # Update filled count
                self.filled_counts[b] = min(
                    self.filled_counts[b] + 1,
                    self.buffer_size
                )
    
    def get_recent(self, num_items: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get most recent items from buffer.
        
        Args:
            num_items: Number of recent items (default: all filled)
            
        Returns:
            keys: [batch, num_items, dim]
            values: [batch, num_items, dim]
        """
        if num_items is None:
            # Return all filled items
            num_items = self.filled_counts.max().item()
        
        num_items = min(num_items, self.buffer_size)
        
        # Get indices of most recent items (accounting for circular nature)
        recent_keys = []
        recent_values = []
        
        for b in range(self.batch_size):
            filled = self.filled_counts[b].item()
            if filled == 0:
                # Empty buffer, return zeros
                recent_keys.append(torch.zeros(num_items, self.dim, device=self.keys_buffer.device))
                recent_values.append(torch.zeros(num_items, self.dim, device=self.values_buffer.device))
            else:
                # Get last num_items items
                items_to_get = min(num_items, filled)
                start_pos = (self.write_positions[b] - items_to_get) % self.buffer_size
                
                if start_pos + items_to_get <= self.buffer_size:
                    # Contiguous range
                    keys_slice = self.keys_buffer[b, start_pos:start_pos + items_to_get]
                    values_slice = self.values_buffer[b, start_pos:start_pos + items_to_get]
                else:
                    # Wraps around
                    first_part = self.buffer_size - start_pos
                    keys_slice = torch.cat([
                        self.keys_buffer[b, start_pos:],
                        self.keys_buffer[b, :items_to_get - first_part]
                    ], dim=0)
                    values_slice = torch.cat([
                        self.values_buffer[b, start_pos:],
                        self.values_buffer[b, :items_to_get - first_part]
                    ], dim=0)
                
                # Pad if needed
                if items_to_get < num_items:
                    pad_size = num_items - items_to_get
                    keys_slice = torch.cat([
                        torch.zeros(pad_size, self.dim, device=keys_slice.device),
                        keys_slice
                    ], dim=0)
                    values_slice = torch.cat([
                        torch.zeros(pad_size, self.dim, device=values_slice.device),
                        values_slice
                    ], dim=0)
                
                recent_keys.append(keys_slice)
                recent_values.append(values_slice)
        
        return torch.stack(recent_keys), torch.stack(recent_values)
    
    def clear(self):
        """Clear all buffers."""
        self.keys_buffer.zero_()
        self.values_buffer.zero_()
        self.write_positions.zero_()
        self.filled_counts.zero_()


class SurpriseDrivenConsolidation(nn.Module):
    """Adaptive consolidation based on cumulative surprise threshold.
    
    Instead of fixed intervals, consolidates when cumulative surprise
    exceeds a threshold, ensuring important memories are promoted.
    """
    
    def __init__(
        self,
        surprise_threshold: float = 5.0,
        medium_threshold: float = 10.0,
        min_steps_between: int = 5
    ):
        super().__init__()
        self.surprise_threshold = surprise_threshold
        self.medium_threshold = medium_threshold
        self.min_steps_between = min_steps_between
        
        # Track cumulative surprise
        self.register_buffer('cumulative_surprise_short', torch.tensor(0.0))
        self.register_buffer('cumulative_surprise_medium', torch.tensor(0.0))
        self.register_buffer('steps_since_short_consolidation', torch.tensor(0))
        self.register_buffer('steps_since_medium_consolidation', torch.tensor(0))
    
    def update_surprise(self, surprise_scores: torch.Tensor):
        """Update cumulative surprise.
        
        Args:
            surprise_scores: [batch, seq_len]
        """
        # Average surprise across batch and sequence
        mean_surprise = surprise_scores.mean()
        
        self.cumulative_surprise_short += mean_surprise
        self.cumulative_surprise_medium += mean_surprise
        self.steps_since_short_consolidation += 1
        self.steps_since_medium_consolidation += 1
    
    def should_consolidate_to_medium(self) -> bool:
        """Check if should consolidate short → medium."""
        return (
            self.cumulative_surprise_short >= self.surprise_threshold and
            self.steps_since_short_consolidation >= self.min_steps_between
        )
    
    def should_consolidate_to_long(self) -> bool:
        """Check if should consolidate medium → long."""
        return (
            self.cumulative_surprise_medium >= self.medium_threshold and
            self.steps_since_medium_consolidation >= self.min_steps_between * 2
        )
    
    def reset_short_timer(self):
        """Reset after short → medium consolidation."""
        self.cumulative_surprise_short = torch.tensor(0.0, device=self.cumulative_surprise_short.device)
        self.steps_since_short_consolidation = torch.tensor(0, device=self.steps_since_short_consolidation.device)
    
    def reset_medium_timer(self):
        """Reset after medium → long consolidation."""
        self.cumulative_surprise_medium = torch.tensor(0.0, device=self.cumulative_surprise_medium.device)
        self.steps_since_medium_consolidation = torch.tensor(0, device=self.steps_since_medium_consolidation.device)


class CrossAttentionRetrieval(nn.Module):
    """Learned cross-attention for memory retrieval.
    
    Uses query transformation and cross-attention instead of
    simple dot-product attention for better retrieval.
    """
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Query transformation
        self.query_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # Multi-head projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(
        self,
        queries: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Cross-attention retrieval.
        
        Args:
            queries: [batch, seq_len, dim]
            memory_keys: [batch, mem_len, dim]
            memory_values: [batch, mem_len, dim]
            return_attention: Whether to return attention weights
            
        Returns:
            retrieved: [batch, seq_len, dim]
            attention_weights: Optional [batch, num_heads, seq_len, mem_len]
        """
        batch_size, seq_len, dim = queries.shape
        mem_len = memory_keys.shape[1]
        
        # Transform queries
        queries = self.query_transform(queries)
        
        # Project to multi-head space
        Q = self.q_proj(queries).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(memory_keys).view(batch_size, mem_len, self.num_heads, self.head_dim)
        V = self.v_proj(memory_values).view(batch_size, mem_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        retrieved = self.out_proj(context)
        
        if return_attention:
            return retrieved, attn_weights
        return retrieved, None


class EarlyStoppingRetrieval(nn.Module):
    """Early stopping for hierarchical retrieval based on confidence.
    
    If retrieval confidence from early tiers is high enough,
    skip later tiers for efficiency.
    """
    
    def __init__(
        self,
        short_term_confidence_threshold: float = 0.9,
        medium_term_confidence_threshold: float = 0.8
    ):
        super().__init__()
        self.short_threshold = short_term_confidence_threshold
        self.medium_threshold = medium_term_confidence_threshold
    
    def compute_confidence(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute retrieval confidence from attention weights.
        
        Args:
            attention_weights: [batch, num_heads, seq_len, mem_len]
            
        Returns:
            confidence: [batch] - confidence score per batch item
        """
        # Maximum attention weight indicates focused retrieval
        max_attn = attention_weights.max(dim=-1)[0]  # [batch, num_heads, seq_len]
        
        # Average across heads and sequence
        confidence = max_attn.mean(dim=(1, 2))  # [batch]
        
        return confidence
    
    def should_skip_medium(self, short_term_confidence: torch.Tensor) -> bool:
        """Check if should skip medium-term retrieval."""
        return (short_term_confidence > self.short_threshold).all().item()
    
    def should_skip_long(self, medium_term_confidence: torch.Tensor) -> bool:
        """Check if should skip long-term retrieval."""
        return (medium_term_confidence > self.medium_threshold).all().item()


class ImportanceDecay(nn.Module):
    """Exponential importance decay for old memories.
    
    Memories gradually lose importance over time, allowing
    room for new important memories.
    """
    
    def __init__(self, decay_rate: float = 0.999):
        super().__init__()
        self.decay_rate = decay_rate
    
    def apply_decay(self, importance: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Apply exponential decay based on age.
        
        Args:
            importance: [batch, mem_len] - current importance scores
            timesteps: [batch, mem_len] - age of each memory
            
        Returns:
            decayed_importance: [batch, mem_len]
        """
        # Exponential decay: importance * decay_rate^age
        max_timestep = timesteps.max()
        age = max_timestep - timesteps
        decay_factor = self.decay_rate ** age
        
        return importance * decay_factor


class MemoryDiversityLoss(nn.Module):
    """Auxiliary loss to encourage diverse memory usage.
    
    Penalizes models that only use a small subset of memory slots,
    encouraging better coverage and utilization.
    """
    
    def __init__(self, diversity_weight: float = 0.01):
        super().__init__()
        self.diversity_weight = diversity_weight
    
    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss from attention patterns.
        
        Args:
            attention_weights: [batch, num_heads, seq_len, mem_len]
            
        Returns:
            loss: Scalar diversity loss
        """
        # Average attention across queries (batch, heads, seq)
        avg_attn = attention_weights.mean(dim=(0, 1, 2))  # [mem_len]
        
        # Compute entropy of memory usage distribution
        # High entropy = diverse usage, low entropy = concentrated
        entropy = -(avg_attn * torch.log(avg_attn + 1e-10)).sum()
        
        # Maximum entropy (uniform distribution)
        mem_len = attention_weights.shape[-1]
        max_entropy = -math.log(1.0 / mem_len)
        
        # Normalize and invert (we want to maximize entropy, so minimize negative entropy)
        normalized_entropy = entropy / max_entropy
        
        # Loss is negative entropy (minimize to maximize diversity)
        loss = self.diversity_weight * (1.0 - normalized_entropy)
        
        return loss


class EpisodicBoundaryDetector(nn.Module):
    """Detect episodic boundaries (document/topic shifts) for consolidation.
    
    Monitors attention patterns and hidden state changes to identify
    when a new episode begins, triggering immediate consolidation.
    """
    
    def __init__(
        self,
        detection_threshold: float = 0.3,
        window_size: int = 10
    ):
        super().__init__()
        self.detection_threshold = detection_threshold
        self.window_size = window_size
        
        # Store recent hidden states for comparison
        self.recent_hidden_states = []
    
    def detect_boundary(
        self,
        current_hidden: torch.Tensor,
        attention_pattern: Optional[torch.Tensor] = None
    ) -> bool:
        """Detect if current position is an episodic boundary.
        
        Args:
            current_hidden: [batch, seq_len, dim]
            attention_pattern: Optional [batch, num_heads, seq_len, seq_len]
            
        Returns:
            is_boundary: Whether boundary detected
        """
        # Method 1: Hidden state discontinuity
        if len(self.recent_hidden_states) >= self.window_size:
            # Compare current with recent average
            recent_avg = torch.stack(self.recent_hidden_states[-self.window_size:]).mean(dim=0)
            
            # Cosine similarity
            similarity = F.cosine_similarity(
                current_hidden.flatten(0, 1),
                recent_avg.flatten(0, 1),
                dim=-1
            ).mean()
            
            # Low similarity = topic shift
            if similarity < (1.0 - self.detection_threshold):
                self.recent_hidden_states.clear()
                return True
        
        # Method 2: Attention pattern change (if provided)
        if attention_pattern is not None:
            # Check if attention is mostly on recent tokens (new topic)
            # vs spread out (continuation)
            avg_attn = attention_pattern.mean(dim=(0, 1))  # [seq_len, seq_len]
            
            # Measure how much attention goes to last 20% of sequence
            seq_len = avg_attn.shape[0]
            recent_portion = int(seq_len * 0.2)
            
            if recent_portion > 0:
                recent_attn_mass = avg_attn[:, -recent_portion:].sum() / seq_len
                
                # If attention is heavily focused on recent tokens
                if recent_attn_mass > 0.7:
                    return True
        
        # Store current state
        self.recent_hidden_states.append(current_hidden.detach().clone())
        if len(self.recent_hidden_states) > self.window_size * 2:
            self.recent_hidden_states.pop(0)
        
        return False
    
    def reset(self):
        """Reset detector state."""
        self.recent_hidden_states.clear()
