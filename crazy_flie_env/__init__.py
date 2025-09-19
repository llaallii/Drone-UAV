# crazy_flie_env/__init__.py
"""
CrazyFlie Reinforcement Learning Environment

A modular, physics-based simulation environment for training autonomous
drone navigation policies using MuJoCo and Gymnasium.

Features:
- Multi-modal observations (state + vision)
- Physics-based PID control
- Advanced camera system
- Configurable reward functions
- Real-time visualization
"""

from .core.environment import CrazyFlieEnv
from .utils.config import EnvConfig, TrainingConfig, TestingConfig

__version__ = "1.0.0"
__all__ = ["CrazyFlieEnv", "EnvConfig", "TrainingConfig", "TestingConfig"]