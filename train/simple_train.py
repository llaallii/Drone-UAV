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
from typing import Tuple, Dict, Any, Optional, Sequence

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import gymnasium as gym

from crazy_flie_env import CrazyFlieEnv
from crazy_flie_env.utils.logging_utils import setup_logging, get_logger, log_system_info, log_training_start, log_training_complete
from .config import TrainingConfig
from .networks import CustomCNN
from .callbacks import RenderCallback, LiveMetricsCallback, DroneStatsCallback

logger = get_logger(__name__)


class ObservationFixWrapper(gym.ObservationWrapper):
    """
    Wrapper to ensure dict observations have correct dtypes for SB3.
    Fixes rollout buffer dtype issues.
    """
    def observation(self, observation):
        if isinstance(observation, dict):
            fixed_obs = {}
            for key, value in observation.items():
                # Ensure proper numpy array with correct dtype
                arr = np.asarray(value)
                if key == 'state':
                    fixed_obs[key] = arr.astype(np.float32)
                elif key == 'image':
                    fixed_obs[key] = arr.astype(np.uint8)
                else:
                    fixed_obs[key] = arr
            return fixed_obs
        return observation


def create_vec_env(config: TrainingConfig):
    """Create vectorized environment using DummyVecEnv for notebook compatibility."""

    def make_env(rank: int):
        def _init():
            env = CrazyFlieEnv(config=config.env_config)
            env = ObservationFixWrapper(env)  # Fix observation dtypes
            env = Monitor(env)
            env.action_space.seed(config.seed + rank)
            return env
        return _init

    env_fns = [make_env(i) for i in range(config.num_envs)]

    # Use DummyVecEnv for notebook compatibility (works on Windows/Jupyter)
    return DummyVecEnv(env_fns)

def setup_directories(config: TrainingConfig) -> Tuple[str, str]:
    """Setup model and log directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/{config.algorithm.lower()}_drone_{timestamp}"
    log_dir = f"logs/{config.algorithm.lower()}_drone_{timestamp}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Log directory: {log_dir}")

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
    # Initialize logging
    setup_logging()
    log_system_info()
    
    if verbose:
        logger.info("Starting PPO Training")
        log_training_start({
            "Algorithm": "PPO",
            "Timesteps": f"{config.total_timesteps:,}",
            "Environments": str(config.num_envs),
            "Device": str(config.device),
            "Learning Rate": str(config.learning_rate),
            "Batch Size": str(config.batch_size)
        })

    # Setup
    model_dir, log_dir = setup_directories(config)
    env = create_vec_env(config)

    # Create eval config with same env_config
    eval_config = TrainingConfig(
        num_envs=1,
        seed=config.seed + 1000,
        env_config=config.env_config
    )
    eval_env = create_vec_env(eval_config)

    # PPO model
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": [256, 256], "vf": [256, 256]}
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

    # Rendering callback (if enabled)
    if config.render_during_training:
        render_callback = RenderCallback(
            render_freq=config.render_freq,
            verbose=1 if verbose else 0
        )
        callbacks.append(render_callback)
        if verbose:
            logger.info(f"Live rendering enabled (every {config.render_freq} steps)")

    # Live metrics callback (if enabled)
    if config.show_live_metrics:
        metrics_callback = LiveMetricsCallback(
            print_freq=config.metrics_freq,
            verbose=1 if verbose else 0
        )
        callbacks.append(metrics_callback)

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
            logger.info(f"Training complete! Model saved to {final_path}")

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
        logger.info("Training interrupted")
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
        logger.info("Starting SAC Training")
        logger.info(f"   Timesteps: {config.total_timesteps:,}")
        logger.info(f"   Environments: {config.num_envs}")
        logger.info(f"   Device: {config.device}")

    # Setup
    model_dir, log_dir = setup_directories(config)
    env = create_vec_env(config)

    # Create eval config with same env_config
    eval_config = TrainingConfig(
        num_envs=1,
        seed=config.seed + 1000,
        env_config=config.env_config
    )
    eval_env = create_vec_env(eval_config)

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

    # Rendering callback (if enabled)
    if config.render_during_training:
        render_callback = RenderCallback(
            render_freq=config.render_freq,
            verbose=1 if verbose else 0
        )
        callbacks.append(render_callback)
        if verbose:
            logger.info(f"Live rendering enabled (every {config.render_freq} steps)")

    # Live metrics callback (if enabled)
    if config.show_live_metrics:
        metrics_callback = LiveMetricsCallback(
            print_freq=config.metrics_freq,
            verbose=1 if verbose else 0
        )
        callbacks.append(metrics_callback)

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
            logger.info(f"Training complete! Model saved to {final_path}")

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
        logger.info("Training interrupted")
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
