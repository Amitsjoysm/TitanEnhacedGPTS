"""Model configurations for Titan-GPT V3.

Production-ready configurations with all V3 enhancements:
- Configurable compression ratios
- Adaptive consolidation settings
- Early stopping thresholds
- Memory diversity weights
- Episodic boundary detection parameters
"""

from typing import Dict


# Small model configuration (124M parameters)
TITAN_V3_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "batch_size": 4,
    
    # V3 Memory configurations
    "use_memory": True,
    "memory_variant": "mac",  # "mac", "mag", "hybrid"
    
    # Hierarchical memory sizes
    "short_term_size": 128,   # GPU tensor buffer
    "medium_term_size": 512,  # Episode memory
    "long_term_size": 2048,   # Semantic memory (compressed)
    
    # V3 Memory features
    "num_memory_layers": 2,
    "surprise_momentum": 0.9,
    
    # Issue #4 fix: Configurable compression
    "compression_ratio": 4,  # 4x compression (can be 2, 4, 8)
    "use_compression": True,
    
    # Issue #8 fix: Cross-attention retrieval
    "num_retrieval_heads": 8,
    "use_1d_conv": True,
    
    # Issue #2 & #6 fix: Adaptive consolidation
    "surprise_threshold_short": 5.0,  # Cumulative surprise for short→medium
    "surprise_threshold_medium": 10.0,  # Cumulative surprise for medium→long
    "min_steps_between_consolidation": 5,
    
    # Issue #3 fix: Early stopping thresholds
    "short_confidence_threshold": 0.9,
    "medium_confidence_threshold": 0.8,
    
    # Issue #7 fix: Importance decay
    "importance_decay_rate": 0.999,
    
    # Issue #9 fix: Memory diversity
    "diversity_loss_weight": 0.01,
    
    # Issue #10 fix: Episodic boundaries
    "boundary_detection_threshold": 0.3,
    "boundary_window_size": 10,
}


# Medium model configuration (340M parameters)
TITAN_V3_CONFIG_340M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "batch_size": 2,
    
    # V3 Memory configurations
    "use_memory": True,
    "memory_variant": "mac",
    
    # Larger memory for medium model
    "short_term_size": 256,
    "medium_term_size": 1024,
    "long_term_size": 4096,
    
    # V3 features
    "num_memory_layers": 3,
    "surprise_momentum": 0.9,
    
    # Configurable compression
    "compression_ratio": 4,
    "use_compression": True,
    
    # Cross-attention
    "num_retrieval_heads": 16,
    "use_1d_conv": True,
    
    # Adaptive consolidation (adjusted for larger model)
    "surprise_threshold_short": 7.0,
    "surprise_threshold_medium": 15.0,
    "min_steps_between_consolidation": 8,
    
    # Early stopping
    "short_confidence_threshold": 0.9,
    "medium_confidence_threshold": 0.8,
    
    # Importance decay
    "importance_decay_rate": 0.999,
    
    # Memory diversity
    "diversity_loss_weight": 0.01,
    
    # Episodic boundaries
    "boundary_detection_threshold": 0.3,
    "boundary_window_size": 15,
}


# Baseline (no memory) for comparison
BASELINE_V3_GPT_124M = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "batch_size": 4,
    "use_memory": False,
}

BASELINE_V3_GPT_340M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "batch_size": 2,
    "use_memory": False,
}


# Training configurations for V3
TRAINING_V3_CONFIG_SMALL = {
    "learning_rate": 5e-4,
    "weight_decay": 0.1,
    "batch_size": 4,
    "num_epochs": 10,
    "warmup_steps": 100,
    "eval_freq": 50,
    "eval_iter": 10,
    "gradient_accumulation_steps": 4,
    
    # V3 specific
    "diversity_loss_weight": 0.01,  # Weight for memory diversity loss
    "use_memory_regularization": True,
    "consolidation_strategy": "adaptive",  # "adaptive", "episodic", "hybrid"
}

TRAINING_V3_CONFIG_MEDIUM = {
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "batch_size": 2,
    "num_epochs": 10,
    "warmup_steps": 200,
    "eval_freq": 100,
    "eval_iter": 20,
    "gradient_accumulation_steps": 8,
    
    # V3 specific
    "diversity_loss_weight": 0.01,
    "use_memory_regularization": True,
    "consolidation_strategy": "adaptive",
}


# Reasoning task configurations for critical thinking evaluation
REASONING_CONFIG = {
    "arithmetic": {
        "num_steps": 3,
        "temperature": 0.7,
        "max_tokens_per_step": 50,
    },
    "logic": {
        "num_steps": 4,
        "temperature": 0.8,
        "max_tokens_per_step": 60,
    },
    "multi_hop": {
        "num_steps": 5,
        "temperature": 0.75,
        "max_tokens_per_step": 70,
    },
}


def get_model_config_v3(
    size: str = "small",
    use_memory: bool = True,
    compression_ratio: int = None
) -> Dict:
    """Get V3 model configuration.
    
    Args:
        size: "small" (124M) or "medium" (340M)
        use_memory: Whether to use V3 memory
        compression_ratio: Override default compression (2, 4, or 8)
        
    Returns:
        Model configuration dict
    """
    if not use_memory:
        configs = {
            "small": BASELINE_V3_GPT_124M,
            "medium": BASELINE_V3_GPT_340M,
        }
    else:
        configs = {
            "small": TITAN_V3_CONFIG_124M,
            "medium": TITAN_V3_CONFIG_340M,
        }
    
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")
    
    config = configs[size].copy()
    
    # Override compression ratio if specified
    if compression_ratio is not None and use_memory:
        if compression_ratio not in [2, 4, 8]:
            raise ValueError("compression_ratio must be 2, 4, or 8")
        config["compression_ratio"] = compression_ratio
    
    return config


def get_training_config_v3(size: str = "small") -> Dict:
    """Get V3 training configuration.
    
    Args:
        size: "small" or "medium"
        
    Returns:
        Training configuration dict
    """
    configs = {
        "small": TRAINING_V3_CONFIG_SMALL,
        "medium": TRAINING_V3_CONFIG_MEDIUM,
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")
    
    return configs[size].copy()


def get_reasoning_config(task_type: str = "arithmetic") -> Dict:
    """Get reasoning task configuration for critical thinking.
    
    Args:
        task_type: "arithmetic", "logic", or "multi_hop"
        
    Returns:
        Reasoning configuration dict
    """
    if task_type not in REASONING_CONFIG:
        raise ValueError(
            f"Unknown task_type '{task_type}'. "
            f"Choose from: {list(REASONING_CONFIG.keys())}"
        )
    
    return REASONING_CONFIG[task_type].copy()
