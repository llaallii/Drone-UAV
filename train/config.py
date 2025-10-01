"""
Simple training configuration for notebook use.
"""

from dataclasses import dataclass
import torch

try:
    from crazy_flie_env.utils.config import EnvConfig
except ImportError:
    EnvConfig = None


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults for research."""

    # Algorithm
    algorithm: str = "PPO"

    # Training duration
    total_timesteps: int = 500_000

    # Environment
    num_envs: int = 4
    seed: int = 42
    env_config: any = None  # Will use EnvConfig() default

    # Learning parameters
    learning_rate: float = 3e-4
    batch_size: int = 64

    # PPO specific
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # SAC specific
    buffer_size: int = 100_000
    tau: float = 0.005

    # Evaluation and checkpointing
    eval_freq: int = 25_000
    save_freq: int = 50_000
    n_eval_episodes: int = 5

    # Device
    device: str = "auto"

    def __post_init__(self):
        """Initialize after creation."""
        # Set environment config
        if self.env_config is None and EnvConfig is not None:
            self.env_config = EnvConfig()

        # Auto-detect device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
