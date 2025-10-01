# ================================
# crazy_flie_env/utils/__init__.py
"""Utility functions and configuration."""

from .config import EnvConfig, TrainingConfig, TestingConfig
from .math_utils import (
    quat_to_euler, euler_to_quat, rotate_vector_by_quat,
    normalize_angle, smooth_interpolate, clamp
)
from .logging_utils import setup_logging, get_logger, log_system_info, log_training_start, log_training_complete

__all__ = [
    "EnvConfig", "TrainingConfig", "TestingConfig",
    "quat_to_euler", "euler_to_quat", "rotate_vector_by_quat",
    "normalize_angle", "smooth_interpolate", "clamp",
    "setup_logging", "get_logger", "log_system_info", "log_training_start", "log_training_complete"
]