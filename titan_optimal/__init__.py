"""Titan-Optimal: Next-Generation LLM Architecture.

Combines Titans architecture with compute-optimal sampling.
"""

__version__ = "1.0.0"
__author__ = "Based on research by Behrouz et al. and Bansal et al."

from .models.titan_gpt import TitanGPTModel
from .models.neural_memory import NeuralMemory
from .configs.model_configs import (
    get_model_config,
    get_training_config,
    get_strategy_config,
)

__all__ = [
    "TitanGPTModel",
    "NeuralMemory",
    "get_model_config",
    "get_training_config",
    "get_strategy_config",
]
