# train_drone_refactored.py
"""
Modular and Flexible Drone Training System

Features:
- Support for multiple RL algorithms (PPO, SAC, DQN, Custom)
- Live training visualization
- Configurable training pipelines
- Algorithm-agnostic design
- Real-time performance monitoring
- Custom algorithm integration
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Union, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import json
import argparse
import traceback
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import environment
try:
    from crazy_flie_env import CrazyFlieEnv
    from crazy_flie_env.utils.config import EnvConfig
    print("✅ Successfully imported CrazyFlieEnv")
except ImportError as e:
    print(f"❌ Error importing CrazyFlieEnv: {e}")
    sys.exit(1)

# Import RL libraries
try:
    from stable_baselines3 import PPO, SAC, DQN, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    print("✅ Stable Baselines3 imported successfully")
except ImportError as e:
    print(f"❌ Error importing Stable Baselines3: {e}")
    print("Install with: pip install stable-baselines3[extra]")
    sys.exit(1)

# Import visualization libraries
try:
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("⚠️ Matplotlib not available - some visualizations disabled")


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
    env_config: Optional[EnvConfig] = None
    
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
    render_freq: int = 1000  # Steps between renders
    
    # Logging
    log_tensorboard: bool = True
    save_replay_buffer: bool = False
    
    def __post_init__(self):
        if self.env_config is None:
            self.env_config = EnvConfig()


class BaseTrainingAlgorithm(ABC):
    """Abstract base class for training algorithms."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.env = None
        
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


class CustomAlgorithmTrainer(BaseTrainingAlgorithm):
    """Template for custom algorithm implementation."""
    
    def __init__(self, config: TrainingConfig, custom_algorithm_class: Type = None):
        super().__init__(config)
        self.custom_algorithm_class = custom_algorithm_class
        
    def create_model(self, env):
        """Create custom model."""
        if self.custom_algorithm_class is None:
            raise ValueError("Custom algorithm class must be provided")
        
        # This is a template - customize based on your algorithm
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


class CustomCNN(BaseFeaturesExtractor):
    """Custom CNN feature extractor for visual inputs."""
    
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        if hasattr(observation_space, 'spaces'):
            # Dict observation space
            img_shape = observation_space.spaces['image'].shape
            state_dim = observation_space.spaces['state'].shape[0]
        else:
            raise ValueError("Expected Dict observation space")
        
        print(f"🧠 CNN Input - Image shape: {img_shape}, State dim: {state_dim}")
        
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(img_shape[2], 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_input = torch.zeros(1, img_shape[2], img_shape[0], img_shape[1])
            cnn_output = self.cnn(sample_input)
            cnn_output_size = cnn_output.view(1, -1).shape[1]
            print(f"🧠 CNN output size: {cnn_output_size}")
        
        # State processing
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(cnn_output_size + 64, features_dim),
            nn.LeakyReLU(),
            nn.Linear(features_dim, features_dim)
        )
        
    def forward(self, observations):
        """Forward pass through the network."""
        try:
            # Process image
            image = observations['image'].float() / 255.0
            
            # Handle batch dimensions
            if len(image.shape) == 4:
                image = image.permute(0, 3, 1, 2)
            elif len(image.shape) == 3:
                image = image.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
            
            # CNN processing
            cnn_features = self.cnn(image).reshape(image.shape[0], -1)
            
            # State processing
            state_features = self.state_processor(observations['state'])
            
            # Feature fusion
            combined = torch.cat([cnn_features, state_features], dim=1)
            return self.fusion(combined)
            
        except Exception as e:
            print(f"❌ Error in CustomCNN forward pass: {e}")
            raise


class LiveTrainingVisualizationCallback(BaseCallback):
    """Callback for live training visualization."""
    
    def __init__(self, env, render_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.render_freq = render_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Setup matplotlib for live plotting
        if matplotlib_available:
            plt.ion()
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
            self.ax1.set_title('Episode Rewards')
            self.ax1.set_xlabel('Episode')
            self.ax1.set_ylabel('Reward')
            
            self.ax2.set_title('Training Progress')
            self.ax2.set_xlabel('Steps')
            self.ax2.set_ylabel('Value')
            
            plt.tight_layout()
            plt.show()
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Track episode rewards
        if 'reward' in self.locals:
            self.current_episode_reward += self.locals['rewards'][0]
        
        # Check for episode end
        if 'dones' in self.locals and self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            self.episode_count += 1
            
            # Update plots
            if matplotlib_available and len(self.episode_rewards) > 0:
                self._update_plots()
        
        # Render environment
        if self.num_timesteps % self.render_freq == 0:
            try:
                # Render the first environment
                if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
                    self.env.envs[0].render()
                elif hasattr(self.env, 'render'):
                    self.env.render()
            except Exception as e:
                print(f"⚠️ Rendering failed: {e}")
        
        return True
    
    def _update_plots(self):
        """Update live training plots."""
        try:
            # Update episode rewards plot
            self.ax1.clear()
            self.ax1.plot(self.episode_rewards, 'b-', alpha=0.7)
            if len(self.episode_rewards) >= 10:
                # Moving average
                window = min(10, len(self.episode_rewards))
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                self.ax1.plot(range(window-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2)
            
            self.ax1.set_title(f'Episode Rewards (Episode {self.episode_count})')
            self.ax1.set_xlabel('Episode')
            self.ax1.set_ylabel('Reward')
            self.ax1.grid(True, alpha=0.3)
            
            # Update training progress
            self.ax2.clear()
            self.ax2.plot(range(len(self.episode_rewards)), self.episode_rewards, 'g-', alpha=0.5)
            self.ax2.set_title(f'Training Progress (Step {self.num_timesteps})')
            self.ax2.set_xlabel('Episode')
            self.ax2.set_ylabel('Reward')
            self.ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(0.01)  # Small pause to update plots
            
        except Exception as e:
            print(f"⚠️ Plot update failed: {e}")


class EnhancedRewardWrapper:
    """Enhanced reward wrapper with configurable reward components."""
    
    def __init__(self, env, reward_config: Dict[str, float] = None):
        self.env = env
        self.reward_config = reward_config or {
            'distance_progress': 10.0,
            'altitude_error': 0.5,
            'stability': 0.1,
            'efficiency': 0.01,
            'collision': -100.0,
            'goal_reached': 100.0
        }
        
        # Episode tracking
        self.prev_distance_to_goal = None
        self.goal_position = None
        self.episode_step = 0
        
    def reset(self, **kwargs):
        """Reset environment and tracking variables."""
        obs, info = self.env.reset(**kwargs)
        
        # Set random goal
        self.goal_position = np.array([
            np.random.uniform(-3.0, 8.0),
            np.random.uniform(-3.0, 8.0),
            np.random.uniform(1.0, 2.5)
        ])
        
        # Reset tracking
        if 'state' in obs:
            current_position = obs['state'][0:3]
            self.prev_distance_to_goal = np.linalg.norm(self.goal_position - current_position)
        
        self.episode_step = 0
        info['goal_position'] = self.goal_position
        
        return obs, info
    
    def step(self, action):
        """Step environment with enhanced reward calculation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1
        
        # Enhanced reward calculation
        enhanced_reward = self._calculate_enhanced_reward(obs, action, terminated, info)
        
        # Add goal info
        info['goal_position'] = self.goal_position
        info['enhanced_reward'] = enhanced_reward
        info['original_reward'] = reward
        
        return obs, enhanced_reward, terminated, truncated, info
    
    def _calculate_enhanced_reward(self, obs, action, terminated, info):
        """Calculate enhanced reward with multiple components."""
        if 'state' not in obs:
            return 0.0
        
        state = obs['state']
        current_position = state[0:3]
        velocity = state[3:6]
        angular_velocity = state[9:12]
        
        total_reward = 0.0
        
        # Distance progress reward
        if self.prev_distance_to_goal is not None:
            current_distance = np.linalg.norm(self.goal_position - current_position)
            progress = self.prev_distance_to_goal - current_distance
            total_reward += progress * self.reward_config['distance_progress']
            self.prev_distance_to_goal = current_distance
            
            # Goal reached bonus
            if current_distance < 0.5:
                total_reward += self.reward_config['goal_reached']
                terminated = True
        
        # Altitude error penalty
        target_altitude = 1.5
        altitude_error = abs(current_position[2] - target_altitude)
        total_reward -= altitude_error * self.reward_config['altitude_error']
        
        # Stability reward
        angular_speed = np.linalg.norm(angular_velocity)
        total_reward -= angular_speed * self.reward_config['stability']
        
        # Efficiency penalty
        action_magnitude = np.linalg.norm(action)
        total_reward -= action_magnitude * self.reward_config['efficiency']
        
        # Collision penalty
        if info.get('is_crashed', False):
            total_reward += self.reward_config['collision']
        
        return total_reward


class TrainingManager:
    """Main training manager that orchestrates the entire training process."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trainer = None
        self.env = None
        self.eval_env = None
        self.model_dir = None
        self.log_dir = None
        
        # Setup directories
        self._setup_directories()
        
        # Initialize trainer based on algorithm
        self._initialize_trainer()
    
    def _setup_directories(self):
        """Setup training and logging directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = f"models/{self.config.algorithm.lower()}_drone_{timestamp}"
        self.log_dir = f"logs/{self.config.algorithm.lower()}_drone_{timestamp}"
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        print(f"📁 Model directory: {self.model_dir}")
        print(f"📁 Log directory: {self.log_dir}")
        
        # Save configuration
        config_dict = {
            'algorithm': self.config.algorithm,
            'total_timesteps': self.config.total_timesteps,
            'learning_rate': self.config.learning_rate,
            'num_envs': self.config.num_envs,
            'enable_live_training': self.config.enable_live_training
        }
        
        with open(os.path.join(self.model_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _initialize_trainer(self):
        """Initialize the appropriate trainer based on algorithm."""
        algorithm_map = {
            'PPO': PPOTrainer,
            'SAC': SACTrainer,
            'DQN': lambda config: self._create_dqn_trainer(config),
            'A2C': lambda config: self._create_a2c_trainer(config),
            'CUSTOM': CustomAlgorithmTrainer
        }
        
        trainer_class = algorithm_map.get(self.config.algorithm.upper())
        if trainer_class is None:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        self.trainer = trainer_class(self.config)
        print(f"🤖 Initialized {self.config.algorithm} trainer")
    
    def _create_dqn_trainer(self, config):
        """Create DQN trainer (placeholder for future implementation)."""
        print("⚠️ DQN trainer not yet implemented")
        return PPOTrainer(config)  # Fallback to PPO
    
    def _create_a2c_trainer(self, config):
        """Create A2C trainer (placeholder for future implementation)."""
        print("⚠️ A2C trainer not yet implemented")
        return PPOTrainer(config)  # Fallback to PPO
    
    def create_environments(self):
        """Create training and evaluation environments."""
        print(f"🏗️ Creating {self.config.num_envs} training environments...")
        
        def make_env(rank: int, seed: int = 0):
            def _init():
                try:
                    # Create base environment
                    env = CrazyFlieEnv(config=self.config.env_config)
                    
                    # Apply reward wrapper
                    env = EnhancedRewardWrapper(env)
                    
                    # Monitor wrapper
                    env = Monitor(env)
                    
                    # Set seed
                    env.reset(seed=seed + rank)
                    
                    return env
                except Exception as e:
                    print(f"❌ Error creating environment {rank}: {e}")
                    raise
            
            set_random_seed(seed)
            return _init
        
        # Create training environments
        if self.config.num_envs > 1:
            self.env = SubprocVecEnv([
                make_env(i, self.config.seed) for i in range(self.config.num_envs)
            ])
        else:
            self.env = DummyVecEnv([make_env(0, self.config.seed)])
        
        # Create evaluation environment
        self.eval_env = DummyVecEnv([make_env(99, self.config.seed)])
        
        print("✅ Environments created successfully")
    
    def create_callbacks(self):
        """Create training callbacks."""
        callbacks = []
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=self.model_dir,
            log_path=self.log_dir,
            eval_freq=self.config.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=self.config.n_eval_episodes
        )
        callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=self.model_dir,
            name_prefix='checkpoint'
        )
        callbacks.append(checkpoint_callback)
        
        # Live training visualization
        if self.config.enable_live_training:
            live_callback = LiveTrainingVisualizationCallback(
                self.env,
                render_freq=self.config.render_freq
            )
            callbacks.append(live_callback)
            print("🎥 Live training visualization enabled")
        
        # Algorithm-specific callbacks
        algo_callbacks = self.trainer.get_algorithm_specific_callbacks()
        callbacks.extend(algo_callbacks)
        
        return callbacks
    
    def train(self):
        """Execute the training process."""
        print(f"🚀 Starting {self.config.algorithm} training...")
        print(f"📊 Total timesteps: {self.config.total_timesteps:,}")
        print(f"🔧 Learning rate: {self.config.learning_rate}")
        print(f"🏗️ Environments: {self.config.num_envs}")
        print("=" * 60)
        
        try:
            # Create environments
            self.create_environments()
            
            # Setup tensorboard logging
            if self.config.log_tensorboard:
                self.trainer.model.set_logger(
                    SummaryWriter(self.log_dir)
                )
            
            # Create callbacks
            callbacks = self.create_callbacks()
            
            # Train the model
            model = self.trainer.train(self.env, callbacks)
            
            # Save final model
            final_model_path = os.path.join(self.model_dir, "final_model")
            model.save(final_model_path)
            print(f"🎉 Training completed! Model saved to {final_model_path}")
            
            return model, self.model_dir
            
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
            try:
                interrupted_path = os.path.join(self.model_dir, "interrupted_model")
                if self.trainer.model is not None:
                    self.trainer.model.save(interrupted_path)
                    print(f"💾 Interrupted model saved to {interrupted_path}")
            except:
                print("⚠️ Could not save interrupted model")
            return self.trainer.model, self.model_dir
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            traceback.print_exc()
            return None, None
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.env is not None:
                self.env.close()
            if self.eval_env is not None:
                self.eval_env.close()
        except:
            pass


def test_trained_model(model_path: str, algorithm: str = "PPO", num_episodes: int = 10):
    """Test a trained model."""
    print(f"🧪 Testing {algorithm} model: {model_path}")
    
    try:
        # Load model
        algorithm_map = {
            'PPO': PPO,
            'SAC': SAC,
            'DQN': DQN,
            'A2C': A2C
        }
        
        model_class = algorithm_map.get(algorithm.upper(), PPO)
        model = model_class.load(model_path)
        print("✅ Model loaded successfully")
        
        # Create test environment
        env_config = EnvConfig()
        env = CrazyFlieEnv(config=env_config)
        env = EnhancedRewardWrapper(env)
        
        episode_rewards = []
        success_count = 0
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            
            while step_count < 1000:  # Max steps per episode
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Render every few episodes
                if episode < 3:
                    env.render()
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            
            # Check success
            goal_pos = info.get('goal_position', np.array([0, 0, 0]))
            current_pos = obs['state'][0:3] if 'state' in obs else np.array([0, 0, 0])
            distance_to_goal = np.linalg.norm(goal_pos - current_pos)
            
            if distance_to_goal < 0.5 and not info.get('is_crashed', False):
                success_count += 1
                print(f"✅ Episode {episode + 1}: Reward = {episode_reward:.2f} (SUCCESS)")
            else:
                print(f"❌ Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        success_rate = success_count / num_episodes
        avg_reward = np.mean(episode_rewards)
        
        print(f"\n📊 Test Results:")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Success Rate: {success_rate:.2%}")
        print(f"   Reward Std: {np.std(episode_rewards):.2f}")
        
        env.close()
        return avg_reward, success_rate
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        traceback.print_exc()
        return None, None


def interactive_config_builder():
    """Interactive configuration builder."""
    print("\n🛠️ Interactive Training Configuration Builder")
    print("=" * 50)
    
    # Algorithm selection
    algorithms = ['PPO', 'SAC', 'DQN', 'A2C', 'CUSTOM']
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


def create_rapid_irl_trainer(config: TrainingConfig):
    """
    Create a RAPID-style IRL trainer based on the research paper.
    This is a template for implementing inverse reinforcement learning.
    """
    
    class RAPIDTrainer(BaseTrainingAlgorithm):
        """RAPID IRL trainer implementation."""
        
        def __init__(self, config: TrainingConfig):
            super().__init__(config)
            self.expert_data = None
            self.q_network = None
            self.policy_network = None
            
        def create_model(self, env):
            """Create RAPID IRL model components."""
            print("🧠 Creating RAPID IRL model...")
            
            # This is a simplified template - implement based on LS-IQ paper
            # You would need to implement:
            # 1. Q-function network
            # 2. Policy network 
            # 3. Expert data loader
            # 4. IRL loss functions
            
            # Placeholder implementation
            from stable_baselines3 import SAC
            
            policy_kwargs = {
                "features_extractor_class": CustomCNN,
                "features_extractor_kwargs": {"features_dim": 256},
                "net_arch": [256, 256],
                "activation_fn": nn.LeakyReLU,
            }
            
            self.model = SAC(
                "MultiInputPolicy",
                env,
                learning_rate=config.learning_rate,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=config.device
            )
            
            print("⚠️ RAPID IRL trainer is a template - implement LS-IQ algorithm")
            return self.model
        
        def load_expert_data(self, expert_data_path: str):
            """Load expert demonstration data."""
            # Implement expert data loading
            print(f"📚 Loading expert data from {expert_data_path}")
            pass
        
        def train(self, env, callbacks: list = None):
            """Train using inverse reinforcement learning."""
            if self.model is None:
                self.create_model(env)
            
            # Implement IRL training loop here
            # This would include:
            # 1. Learning Q-function from expert and learner data
            # 2. Policy updates using soft actor-critic
            # 3. Handling absorbing states
            # 4. Auxiliary autoencoder loss
            
            # For now, fallback to standard SAC training
            print("🔄 Training with RAPID IRL (simplified implementation)")
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks
            )
            return self.model
        
        def get_algorithm_specific_callbacks(self) -> list:
            """Get RAPID-specific callbacks."""
            return []
    
    return RAPIDTrainer(config)


def benchmark_algorithms():
    """Benchmark multiple algorithms on the same task."""
    print("\n🏁 Algorithm Benchmark Mode")
    print("=" * 40)
    
    algorithms_to_test = ['PPO', 'SAC']
    timesteps_per_algo = 100000  # Shorter for benchmarking
    
    results = {}
    
    for algorithm in algorithms_to_test:
        print(f"\n🤖 Benchmarking {algorithm}...")
        
        config = TrainingConfig(
            algorithm=algorithm,
            total_timesteps=timesteps_per_algo,
            num_envs=2,  # Fewer environments for faster benchmarking
            enable_live_training=False
        )
        
        manager = TrainingManager(config)
        
        try:
            model, model_dir = manager.train()
            
            if model is not None:
                # Test the trained model
                model_path = os.path.join(model_dir, "final_model")
                avg_reward, success_rate = test_trained_model(
                    model_path, algorithm, num_episodes=5
                )
                
                results[algorithm] = {
                    'avg_reward': avg_reward,
                    'success_rate': success_rate,
                    'model_path': model_path
                }
            
        except Exception as e:
            print(f"❌ {algorithm} benchmark failed: {e}")
            results[algorithm] = None
        
        finally:
            manager.cleanup()
    
    # Print benchmark results
    print("\n📊 Benchmark Results:")
    print("=" * 40)
    
    for algo, result in results.items():
        if result is not None:
            print(f"{algo:10} | Avg Reward: {result['avg_reward']:8.2f} | "
                  f"Success Rate: {result['success_rate']:6.1%}")
        else:
            print(f"{algo:10} | FAILED")
    
    return results


def main():
    """Main training function with multiple modes."""
    parser = argparse.ArgumentParser(description="Flexible Drone Navigation Training System")
    
    # Training modes
    parser.add_argument("--mode", type=str, default="interactive", 
                       choices=["interactive", "config", "benchmark", "test"],
                       help="Training mode")
    
    # Algorithm selection
    parser.add_argument("--algorithm", type=str, default="PPO",
                       choices=["PPO", "SAC", "DQN", "A2C", "CUSTOM", "RAPID"],
                       help="RL algorithm to use")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=500_000,
                       help="Total training timesteps")
    parser.add_argument("--num_envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    
    # Visualization
    parser.add_argument("--live_training", action="store_true",
                       help="Enable live training visualization")
    parser.add_argument("--render_freq", type=int, default=1000,
                       help="Rendering frequency during training")
    
    # Testing
    parser.add_argument("--test_model", type=str, default=None,
                       help="Path to model for testing")
    parser.add_argument("--test_episodes", type=int, default=10,
                       help="Number of test episodes")
    
    # Device and seed
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cpu/cuda/auto)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("🚁 Autonomous Drone Navigation Training System")
    print("=" * 50)
    
    # Handle different modes
    if args.mode == "interactive":
        config = interactive_config_builder()
    
    elif args.mode == "benchmark":
        benchmark_algorithms()
        return
    
    elif args.mode == "test":
        if args.test_model is None:
            print("❌ --test_model path required for test mode")
            return
        
        test_trained_model(args.test_model, args.algorithm, args.test_episodes)
        return
    
    else:  # config mode
        config = TrainingConfig(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            learning_rate=args.learning_rate,
            enable_live_training=args.live_training,
            render_freq=args.render_freq,
            device=args.device,
            seed=args.seed
        )
    
    # Ask about live training if not specified
    if not config.enable_live_training and args.mode == "interactive":
        live_prompt = input("\n🎥 Would you like to see live training visualization? [y/N]: ")
        if live_prompt.strip().lower() in ['y', 'yes', '1', 'true']:
            config.enable_live_training = True
            print("✅ Live training visualization enabled!")
            print("   You will see:")
            print("   - Real-time environment rendering")
            print("   - Live reward plots")
            print("   - Training progress graphs")
    
    # Special handling for RAPID algorithm
    if config.algorithm.upper() == "RAPID":
        print("🧠 RAPID IRL Algorithm Selected")
        print("   Based on: 'RAPID: Robust and Agile Planner Using Inverse Reinforcement Learning'")
        
        # Create RAPID trainer
        config.algorithm = "CUSTOM"  # Use custom trainer framework
        manager = TrainingManager(config)
        manager.trainer = create_rapid_irl_trainer(config)
    else:
        # Standard training
        manager = TrainingManager(config)
    
    # Execute training
    print(f"\n🚀 Starting training with {config.algorithm}...")
    if config.enable_live_training:
        print("🎥 Live visualization active - you'll see the drone learning in real-time!")
    
    try:
        result = manager.train()
        
        if result[0] is not None:
            model, model_dir = result
            print(f"\n🧪 Testing trained {config.algorithm} model...")
            
            # Test the trained model
            model_path = os.path.join(model_dir, "final_model")
            test_trained_model(model_path, config.algorithm)
        else:
            print("❌ Training failed - no model to test")
    
    finally:
        manager.cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Training system interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        traceback.print_exc()