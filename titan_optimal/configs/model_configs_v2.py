"""Enhanced model configurations for Titan-GPT v2.

Adds configurations for the enhanced v2 architecture with:
- Hierarchical memory settings
- Memory compression options
- Separate training/inference modes
"""

from typing import Dict


# Enhanced Small model configuration (124M parameters)
TITAN_V2_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    
    # V2 Memory configurations
    "use_memory": True,
    "memory_variant": "mac",  # "mac", "mag", "hybrid"
    
    # Hierarchical memory sizes
    "short_term_size": 128,   # Recent tokens buffer
    "medium_term_size": 512,  # Episode/document memory
    "long_term_size": 2048,   # Cross-document semantic memory
    
    # Memory features
    "num_memory_layers": 2,
    "surprise_momentum": 0.9,
    "use_compression": True,
    "use_1d_conv": True,
}


# Enhanced Medium model configuration (340M parameters)
TITAN_V2_CONFIG_340M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False,
    
    # V2 Memory configurations
    "use_memory": True,
    "memory_variant": "mac",
    
    # Hierarchical memory sizes (larger for medium model)
    "short_term_size": 256,
    "medium_term_size": 1024,
    "long_term_size": 4096,
    
    # Memory features
    "num_memory_layers": 3,
    "surprise_momentum": 0.9,
    "use_compression": True,
    "use_1d_conv": True,
}


# Baseline V2 (no memory) for comparison
BASELINE_V2_GPT_124M = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "use_memory": False,
}

BASELINE_V2_GPT_340M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "use_memory": False,
}


# Training configurations for v2 models
TRAINING_V2_CONFIG_SMALL = {
    "learning_rate": 5e-4,
    "weight_decay": 0.1,
    "batch_size": 4,
    "num_epochs": 10,
    "warmup_steps": 100,
    "eval_freq": 50,
    "eval_iter": 10,
    "gradient_accumulation_steps": 4,
    
    # V2 specific
    "use_curriculum_learning": True,
    "memory_update_schedule": "adaptive",  # "always", "adaptive", "periodic"
}

TRAINING_V2_CONFIG_MEDIUM = {
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "batch_size": 2,
    "num_epochs": 10,
    "warmup_steps": 200,
    "eval_freq": 100,
    "eval_iter": 20,
    "gradient_accumulation_steps": 8,
    
    # V2 specific
    "use_curriculum_learning": True,
    "memory_update_schedule": "adaptive",
}


def get_model_config_v2(size: str = "small", use_memory: bool = True) -> Dict:
    """Get v2 model configuration by size.
    
    Args:
        size: "small" (124M) or "medium" (340M)
        use_memory: Whether to use enhanced memory
        
    Returns:
        Model configuration dict
    """
    if not use_memory:
        configs = {
            "small": BASELINE_V2_GPT_124M,
            "medium": BASELINE_V2_GPT_340M,
        }
    else:
        configs = {
            "small": TITAN_V2_CONFIG_124M,
            "medium": TITAN_V2_CONFIG_340M,
        }
    
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")
    
    return configs[size].copy()


def get_training_config_v2(size: str = "small") -> Dict:
    """Get v2 training configuration by size.
    
    Args:
        size: "small" or "medium"
        
    Returns:
        Training configuration dict
    """
    configs = {
        "small": TRAINING_V2_CONFIG_SMALL,
        "medium": TRAINING_V2_CONFIG_MEDIUM,
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")
    
    return configs[size].copy()
