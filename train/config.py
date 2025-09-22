# train/config.py
"""
Training configuration management.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch

try:
    from crazy_flie_env.utils.config import EnvConfig
except ImportError:
    print("⚠️ CrazyFlieEnv not found - using default config")
    EnvConfig = None


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # General training settings
    algorithm: str = "PPO"
    total_timesteps: int = 500_000
    num_envs: int = 4
    seed: int = 42
    device: str = "auto"
    
    # Environment settings
    env_config: Optional[Any] = None
    
    # Algorithm-specific hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 1024
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # SAC specific
    buffer_size: int = 100000
    tau: float = 0.005
    
    # Network architecture
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation and saving
    eval_freq: int = 25000
    save_freq: int = 50000
    n_eval_episodes: int = 5
    
    # Live visualization
    enable_live_training: bool = False
    render_freq: int = 1000
    
    # Logging
    log_tensorboard: bool = True
    save_replay_buffer: bool = False
    
    def __post_init__(self):
        if self.env_config is None and EnvConfig is not None:
            self.env_config = EnvConfig()
        
        # Auto-detect device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


def create_default_configs():
    """Create default configurations for different algorithms."""
    
    configs = {
        "PPO": TrainingConfig(
            algorithm="PPO",
            learning_rate=3e-4,
            batch_size=64,
            n_steps=2048,
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.01
        ),
        
        "SAC": TrainingConfig(
            algorithm="SAC",
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=1_000_000,
            tau=0.005
        ),
        
        "RAPID": TrainingConfig(
            algorithm="RAPID",
            learning_rate=1e-4,
            total_timesteps=2_000_000,
            batch_size=128,
            policy_kwargs={"features_dim": 512}
        )
    }
    
    return configs


def interactive_config_builder():
    """Interactive configuration builder."""
    print("\n🛠️ Interactive Training Configuration Builder")
    print("=" * 50)
    
    # Algorithm selection
    algorithms = ['PPO', 'SAC', 'DQN', 'A2C', 'RAPID', 'CUSTOM']
    print(f"Available algorithms: {', '.join(algorithms)}")
    
    while True:
        algorithm = input("Select algorithm [PPO]: ").strip().upper()
        if not algorithm:
            algorithm = 'PPO'
        if algorithm in algorithms:
            break
        print(f"❌ Invalid algorithm. Choose from: {algorithms}")
    
    # Training parameters
    print(f"\n📊 Training Parameters for {algorithm}")
    
    try:
        timesteps = int(input("Total timesteps [500000]: ") or "500000")
        num_envs = int(input("Number of parallel environments [4]: ") or "4")
        learning_rate = float(input("Learning rate [3e-4]: ") or "3e-4")
    except ValueError:
        print("⚠️ Invalid input, using default values")
        timesteps = 500000
        num_envs = 4
        learning_rate = 3e-4
    
    # Live training visualization
    live_training = input("Enable live training visualization? [y/N]: ").strip().lower()
    enable_live = live_training in ['y', 'yes', '1', 'true']
    
    if enable_live:
        try:
            render_freq = int(input("Render frequency (steps) [1000]: ") or "1000")
        except ValueError:
            render_freq = 1000
    else:
        render_freq = 1000
    
    # Create configuration
    config = TrainingConfig(
        algorithm=algorithm,
        total_timesteps=timesteps,
        num_envs=num_envs,
        learning_rate=learning_rate,
        enable_live_training=enable_live,
        render_freq=render_freq
    )
    
    print(f"\n✅ Configuration created:")
    print(f"   Algorithm: {config.algorithm}")
    print(f"   Timesteps: {config.total_timesteps:,}")
    print(f"   Environments: {config.num_envs}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Live Training: {'✅' if config.enable_live_training else '❌'}")
    
    return config