"""Enhanced GPT with Titan Architecture Integration

This extends the original GPT implementation from ch04/ch05 with Titan's
neural long-term memory capabilities.

Can be used as a drop-in replacement for GPTModel with use_memory flag.
"""

import sys
from pathlib import Path

# Add titan-optimal to path
sys.path.append(str(Path(__file__).parent.parent.parent / "titan-optimal"))

from models.titan_gpt import TitanGPTModel
from configs.model_configs import get_model_config

# For backward compatibility, export as GPTModel
GPTModelTitan = TitanGPTModel


def create_titan_model(base_config: dict, use_memory: bool = True) -> TitanGPTModel:
    """Create a Titan-enhanced GPT model from base config.
    
    Args:
        base_config: Standard GPT configuration dict
        use_memory: Whether to enable neural memory
        
    Returns:
        TitanGPTModel instance
    """
    # Convert base config to Titan config
    titan_config = base_config.copy()
    titan_config["use_memory"] = use_memory
    
    # Add Titan-specific defaults if not present
    titan_config.setdefault("memory_size", 512)
    titan_config.setdefault("memory_variant", "mac")
    titan_config.setdefault("num_memory_layers", 2)
    titan_config.setdefault("surprise_momentum", 0.9)
    titan_config.setdefault("forget_decay", 0.01)
    titan_config.setdefault("use_1d_conv", True)
    
    return TitanGPTModel(titan_config)


# Example usage
if __name__ == "__main__":
    import torch
    
    # Standard GPT-124M config
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    # Create Titan-enhanced model
    print("Creating Titan-enhanced GPT...")
    model = create_titan_model(GPT_CONFIG_124M, use_memory=True)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    dummy_input = torch.randint(0, 50257, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(dummy_input, update_memory=False)
    
    print(f"Output shape: {output.shape}")
    print("Titan-enhanced GPT working correctly!")
