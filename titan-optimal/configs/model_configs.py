"""Configuration presets for different model sizes and training strategies."""

from typing import Dict


# Small model configuration (124M parameters)
# Good for CPU training and quick iteration
TITAN_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 512,  # Shorter for memory efficiency
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    
    # Titan-specific configs
    "use_memory": True,
    "memory_variant": "mac",  # Options: "mac", "mag", "hybrid"
    "memory_size": 512,  # Size of long-term memory
    "num_memory_layers": 2,  # Depth of memory MLP
    "surprise_momentum": 0.9,
    "forget_decay": 0.01,
    "use_1d_conv": True,
}


# Medium model configuration (340M parameters)
# Similar to sizes tested in Titans paper
TITAN_CONFIG_340M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False,
    
    # Titan-specific configs
    "use_memory": True,
    "memory_variant": "mac",
    "memory_size": 1024,
    "num_memory_layers": 2,
    "surprise_momentum": 0.9,
    "forget_decay": 0.01,
    "use_1d_conv": True,
}


# Large model configuration (760M parameters)
# Upper range from Titans paper
TITAN_CONFIG_760M = {
    "vocab_size": 50257,
    "context_length": 2048,
    "emb_dim": 1280,
    "n_heads": 20,
    "n_layers": 36,
    "drop_rate": 0.1,
    "qkv_bias": False,
    
    # Titan-specific configs
    "use_memory": True,
    "memory_variant": "mac",
    "memory_size": 2048,
    "num_memory_layers": 2,
    "surprise_momentum": 0.9,
    "forget_decay": 0.01,
    "use_1d_conv": True,
}


# Baseline GPT without memory (for comparison)
BASELINE_GPT_124M = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    
    # Disable memory
    "use_memory": False,
}


# Training configurations
TRAINING_CONFIG_SMALL = {
    "learning_rate": 5e-4,
    "weight_decay": 0.1,
    "batch_size": 4,
    "num_epochs": 10,
    "warmup_steps": 100,
    "eval_freq": 50,
    "eval_iter": 10,
    "gradient_accumulation_steps": 4,
    
    # Compute-optimal sampling configs
    "use_synthetic_data": True,
    "samples_per_prompt": 3,  # WC model generates more
    "temperature": 0.7,
    "top_k": 50,
}


TRAINING_CONFIG_MEDIUM = {
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "batch_size": 2,
    "num_epochs": 20,
    "warmup_steps": 200,
    "eval_freq": 100,
    "eval_iter": 20,
    "gradient_accumulation_steps": 8,
    
    # Compute-optimal sampling configs
    "use_synthetic_data": True,
    "samples_per_prompt": 5,
    "temperature": 0.7,
    "top_k": 50,
}


# Compute-optimal strategy configurations
STRATEGY_CONFIGS = {
    "self_improvement": {
        "description": "Model trains on its own generations",
        "requires_teacher": False,
        "samples_multiplier": 3,  # Generate 3x more samples
    },
    "distillation": {
        "description": "Student learns from teacher generations",
        "requires_teacher": True,
        "samples_multiplier": 1,
    },
    "weak_to_strong": {
        "description": "Weaker model teaches stronger model",
        "requires_teacher": True,
        "samples_multiplier": 3,  # Weak model generates more
        "teacher_is_weaker": True,
    },
}


def get_model_config(size: str = "small", use_memory: bool = True) -> Dict:
    """Get model configuration by size.
    
    Args:
        size: "small" (124M), "medium" (340M), or "large" (760M)
        use_memory: Whether to use neural memory (Titan) or baseline GPT
        
    Returns:
        Model configuration dict
    """
    if not use_memory:
        return BASELINE_GPT_124M.copy()
    
    configs = {
        "small": TITAN_CONFIG_124M,
        "medium": TITAN_CONFIG_340M,
        "large": TITAN_CONFIG_760M,
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")
    
    return configs[size].copy()


def get_training_config(size: str = "small") -> Dict:
    """Get training configuration by size.
    
    Args:
        size: "small" or "medium"
        
    Returns:
        Training configuration dict
    """
    configs = {
        "small": TRAINING_CONFIG_SMALL,
        "medium": TRAINING_CONFIG_MEDIUM,
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")
    
    return configs[size].copy()


def get_strategy_config(strategy: str) -> Dict:
    """Get compute-optimal strategy configuration.
    
    Args:
        strategy: "self_improvement", "distillation", or "weak_to_strong"
        
    Returns:
        Strategy configuration dict
    """
    if strategy not in STRATEGY_CONFIGS:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGY_CONFIGS.keys())}"
        )
    
    return STRATEGY_CONFIGS[strategy].copy()
