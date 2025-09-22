# train/algorithms.py
"""
Algorithm implementations and factory.
"""

from abc import ABC, abstractmethod
from typing import Any, Type, Dict, Optional
import torch.nn as nn

try:
    from stable_baselines3 import PPO, SAC, DQN, A2C
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError:
    print("❌ Stable Baselines3 not found. Install with: pip install stable-baselines3[extra]")
    raise

from .config import TrainingConfig
from .networks import CustomCNN


class BaseTrainingAlgorithm(ABC):
    """Abstract base class for training algorithms."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        
    @abstractmethod
    def create_model(self, env) -> Any:
        """Create and return the RL model."""
        pass
    
    @abstractmethod
    def train(self, env, callbacks: list = None) -> Any:
        """Train the model and return it."""
        pass
    
    @abstractmethod
    def get_algorithm_specific_callbacks(self) -> list:
        """Return algorithm-specific callbacks."""
        pass
    
    def get_model_save_path(self, base_dir: str) -> str:
        """Get the path for saving the model."""
        import os
        return os.path.join(base_dir, f"{self.config.algorithm.lower()}_model")


class PPOTrainer(BaseTrainingAlgorithm):
    """PPO algorithm trainer."""
    
    def create_model(self, env):
        """Create PPO model."""
        policy_kwargs = {
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": [{"pi": [256, 256], "vf": [256, 256]}],
            "activation_fn": nn.LeakyReLU,
            **self.config.policy_kwargs
        }
        
        self.model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.config.device
        )
        return self.model
    
    def train(self, env, callbacks: list = None):
        """Train PPO model."""
        if self.model is None:
            self.create_model(env)
        
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks
        )
        return self.model
    
    def get_algorithm_specific_callbacks(self) -> list:
        """Get PPO-specific callbacks."""
        return []


class SACTrainer(BaseTrainingAlgorithm):
    """SAC algorithm trainer."""
    
    def create_model(self, env):
        """Create SAC model."""
        policy_kwargs = {
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": [256, 256],
            "activation_fn": nn.LeakyReLU,
            **self.config.policy_kwargs
        }
        
        self.model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.config.device
        )
        return self.model
    
    def train(self, env, callbacks: list = None):
        """Train SAC model."""
        if self.model is None:
            self.create_model(env)
        
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks
        )
        return self.model
    
    def get_algorithm_specific_callbacks(self) -> list:
        """Get SAC-specific callbacks."""
        return []


class RAPIDTrainer(BaseTrainingAlgorithm):
    """RAPID IRL trainer - template implementation."""
    
    def create_model(self, env):
        """Create RAPID IRL model."""
        print("🧠 Creating RAPID IRL model (template implementation)...")
        
        # Template implementation - replace with actual LS-IQ algorithm
        policy_kwargs = {
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": [512, 512, 256],
            "activation_fn": nn.LeakyReLU,
            **self.config.policy_kwargs
        }
        
        # Use SAC as base for continuous control
        self.model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=self.config.learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.config.device
        )
        
        print("⚠️ RAPID trainer is a template - implement LS-IQ algorithm for full functionality")
        return self.model
    
    def load_expert_data(self, expert_data_path: str):
        """Load expert demonstration data."""
        print(f"📚 Loading expert data from {expert_data_path}")
        # Implement expert data loading for IRL
        pass
    
    def train(self, env, callbacks: list = None):
        """Train using inverse reinforcement learning."""
        if self.model is None:
            self.create_model(env)
        
        print("🔄 Training with RAPID IRL (simplified implementation)")
        # Implement IRL training loop here
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks
        )
        return self.model
    
    def get_algorithm_specific_callbacks(self) -> list:
        """Get RAPID-specific callbacks."""
        return []


class CustomAlgorithmTrainer(BaseTrainingAlgorithm):
    """Template for custom algorithm implementation."""
    
    def __init__(self, config: TrainingConfig, custom_algorithm_class: Type = None):
        super().__init__(config)
        self.custom_algorithm_class = custom_algorithm_class
        
    def create_model(self, env):
        """Create custom model."""
        if self.custom_algorithm_class is None:
            raise ValueError("Custom algorithm class must be provided")
        
        self.model = self.custom_algorithm_class(
            env=env,
            config=self.config
        )
        return self.model
    
    def train(self, env, callbacks: list = None):
        """Train custom model."""
        if self.model is None:
            self.create_model(env)
        
        # Implement custom training loop
        self.model.train(timesteps=self.config.total_timesteps)
        return self.model
    
    def get_algorithm_specific_callbacks(self) -> list:
        """Get custom algorithm callbacks."""
        return []


class AlgorithmFactory:
    """Factory for creating algorithm trainers."""
    
    _algorithm_map: Dict[str, Type[BaseTrainingAlgorithm]] = {
        'PPO': PPOTrainer,
        'SAC': SACTrainer,
        'RAPID': RAPIDTrainer,
        'CUSTOM': CustomAlgorithmTrainer
    }
    
    @classmethod
    def create_trainer(cls, config: TrainingConfig, custom_class: Optional[Type] = None) -> BaseTrainingAlgorithm:
        """Create trainer based on algorithm name."""
        algorithm = config.algorithm.upper()
        
        if algorithm not in cls._algorithm_map:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Available: {list(cls._algorithm_map.keys())}")
        
        trainer_class = cls._algorithm_map[algorithm]
        
        if algorithm == 'CUSTOM' and custom_class is not None:
            return trainer_class(config, custom_class)
        else:
            return trainer_class(config)
    
    @classmethod
    def register_algorithm(cls, name: str, trainer_class: Type[BaseTrainingAlgorithm]):
        """Register a new algorithm trainer."""
        cls._algorithm_map[name.upper()] = trainer_class
        print(f"✅ Registered algorithm: {name}")
    
    @classmethod
    def list_algorithms(cls) -> list:
        """List all available algorithms."""
        return list(cls._algorithm_map.keys())


# Example of registering a custom algorithm
def register_custom_algorithms():
    """Register additional custom algorithms."""
    
    # Example: A2C trainer (simplified)
    class A2CTrainer(BaseTrainingAlgorithm):
        def create_model(self, env):
            policy_kwargs = {
                "features_extractor_class": CustomCNN,
                "features_extractor_kwargs": {"features_dim": 256},
                "net_arch": [256, 256],
                "activation_fn": nn.LeakyReLU,
            }
            
            self.model = A2C(
                "MultiInputPolicy",
                env,
                learning_rate=self.config.learning_rate,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=self.config.device
            )
            return self.model
        
        def train(self, env, callbacks: list = None):
            if self.model is None:
                self.create_model(env)
            self.model.learn(total_timesteps=self.config.total_timesteps, callback=callbacks)
            return self.model
        
        def get_algorithm_specific_callbacks(self) -> list:
            return []
    
    # Register the A2C trainer
    AlgorithmFactory.register_algorithm('A2C', A2CTrainer)


# Auto-register additional algorithms
register_custom_algorithms()