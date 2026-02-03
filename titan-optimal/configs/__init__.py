"""Configs package."""

from .model_configs import (
    TITAN_CONFIG_124M,
    TITAN_CONFIG_340M,
    TITAN_CONFIG_760M,
    BASELINE_GPT_124M,
    TRAINING_CONFIG_SMALL,
    TRAINING_CONFIG_MEDIUM,
    STRATEGY_CONFIGS,
    get_model_config,
    get_training_config,
    get_strategy_config,
)

__all__ = [
    "TITAN_CONFIG_124M",
    "TITAN_CONFIG_340M",
    "TITAN_CONFIG_760M",
    "BASELINE_GPT_124M",
    "TRAINING_CONFIG_SMALL",
    "TRAINING_CONFIG_MEDIUM",
    "STRATEGY_CONFIGS",
    "get_model_config",
    "get_training_config",
    "get_strategy_config",
]
