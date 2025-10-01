"""
Simple training module for Jupyter notebooks.

Usage in notebook:
    from train.simple_train import train_ppo, train_sac, test_model
    from train.config import TrainingConfig

    config = TrainingConfig(total_timesteps=100000)
    model, results = train_ppo(config)
"""

import os
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from crazy_flie_env import CrazyFlieEnv
from .config import TrainingConfig
from .networks import CustomCNN


def create_env(config: TrainingConfig, rank: int = 0):
    """Create a single training environment."""
    def _init():
        env = CrazyFlieEnv(config=config.env_config)
        env = Monitor(env)
        env.reset(seed=config.seed + rank)
        return env
    return _init


def create_vec_env(config: TrainingConfig):
    """Create vectorized environment for parallel training."""
    env_fns = [create_env(config, i) for i in range(config.num_envs)]

    if config.num_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


def setup_directories(config: TrainingConfig) -> Tuple[str, str]:
    """Setup model and log directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/{config.algorithm.lower()}_drone_{timestamp}"
    log_dir = f"logs/{config.algorithm.lower()}_drone_{timestamp}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"📁 Model directory: {model_dir}")
    print(f"📁 Log directory: {log_dir}")

    return model_dir, log_dir


def train_ppo(config: TrainingConfig, verbose: bool = True) -> Tuple[PPO, Dict[str, Any]]:
    """
    Train a PPO agent.

    Args:
        config: Training configuration
        verbose: Print training progress

    Returns:
        Tuple of (trained_model, results_dict)
    """
    if verbose:
        print("🚀 Starting PPO Training")
        print(f"   Timesteps: {config.total_timesteps:,}")
        print(f"   Environments: {config.num_envs}")
        print(f"   Device: {config.device}")

    # Setup
    model_dir, log_dir = setup_directories(config)
    env = create_vec_env(config)
    eval_env = create_vec_env(TrainingConfig(num_envs=1, seed=config.seed + 1000))

    # PPO model
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [{"pi": [256, 256], "vf": [256, 256]}]
    }

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        policy_kwargs=policy_kwargs,
        verbose=1 if verbose else 0,
        device=config.device,
        tensorboard_log=log_dir
    )

    # Callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=model_dir,
        name_prefix='checkpoint'
    )
    callbacks.append(checkpoint_callback)

    # Eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        verbose=1 if verbose else 0
    )
    callbacks.append(eval_callback)

    # Train
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # Save final model
        final_path = os.path.join(model_dir, "final_model")
        model.save(final_path)

        if verbose:
            print(f"✅ Training complete! Model saved to {final_path}")

        # Clean up
        env.close()
        eval_env.close()

        results = {
            'model_dir': model_dir,
            'log_dir': log_dir,
            'final_model_path': final_path,
            'algorithm': 'PPO',
            'total_timesteps': config.total_timesteps
        }

        return model, results

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted")
        interrupted_path = os.path.join(model_dir, "interrupted_model")
        model.save(interrupted_path)
        env.close()
        eval_env.close()

        results = {
            'model_dir': model_dir,
            'interrupted': True,
            'model_path': interrupted_path
        }
        return model, results


def train_sac(config: TrainingConfig, verbose: bool = True) -> Tuple[SAC, Dict[str, Any]]:
    """
    Train a SAC agent.

    Args:
        config: Training configuration
        verbose: Print training progress

    Returns:
        Tuple of (trained_model, results_dict)
    """
    if verbose:
        print("🚀 Starting SAC Training")
        print(f"   Timesteps: {config.total_timesteps:,}")
        print(f"   Environments: {config.num_envs}")
        print(f"   Device: {config.device}")

    # Setup
    model_dir, log_dir = setup_directories(config)
    env = create_vec_env(config)
    eval_env = create_vec_env(TrainingConfig(num_envs=1, seed=config.seed + 1000))

    # SAC model
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [256, 256]
    }

    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        tau=config.tau,
        gamma=config.gamma,
        policy_kwargs=policy_kwargs,
        verbose=1 if verbose else 0,
        device=config.device,
        tensorboard_log=log_dir
    )

    # Callbacks
    callbacks = []

    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=model_dir,
        name_prefix='checkpoint'
    )
    callbacks.append(checkpoint_callback)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        verbose=1 if verbose else 0
    )
    callbacks.append(eval_callback)

    # Train
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # Save final model
        final_path = os.path.join(model_dir, "final_model")
        model.save(final_path)

        if verbose:
            print(f"✅ Training complete! Model saved to {final_path}")

        # Clean up
        env.close()
        eval_env.close()

        results = {
            'model_dir': model_dir,
            'log_dir': log_dir,
            'final_model_path': final_path,
            'algorithm': 'SAC',
            'total_timesteps': config.total_timesteps
        }

        return model, results

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted")
        interrupted_path = os.path.join(model_dir, "interrupted_model")
        model.save(interrupted_path)
        env.close()
        eval_env.close()

        results = {
            'model_dir': model_dir,
            'interrupted': True,
            'model_path': interrupted_path
        }
        return model, results


def load_model(model_path: str, algorithm: str = "PPO"):
    """
    Load a trained model.

    Args:
        model_path: Path to the model file
        algorithm: Algorithm type (PPO or SAC)

    Returns:
        Loaded model
    """
    if algorithm.upper() == "PPO":
        return PPO.load(model_path)
    elif algorithm.upper() == "SAC":
        return SAC.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
