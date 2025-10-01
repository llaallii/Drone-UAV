"""
Utilities for testing trained models in notebooks.
"""

import numpy as np
from typing import Tuple, Optional, List
from stable_baselines3 import PPO, SAC
from crazy_flie_env import CrazyFlieEnv
from crazy_flie_env.utils.config import EnvConfig


def test_model(model_path: str,
               algorithm: str = "PPO",
               num_episodes: int = 10,
               render: bool = True,
               verbose: bool = True) -> Tuple[float, dict]:
    """
    Test a trained model and return performance metrics.

    Args:
        model_path: Path to saved model
        algorithm: Algorithm type (PPO or SAC)
        num_episodes: Number of test episodes
        render: Whether to render episodes
        verbose: Print episode results

    Returns:
        Tuple of (average_reward, metrics_dict)
    """

    # Load model
    if algorithm.upper() == "PPO":
        model = PPO.load(model_path)
    elif algorithm.upper() == "SAC":
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if verbose:
        print(f"🧪 Testing {algorithm} model")
        print(f"   Episodes: {num_episodes}")
        print(f"   Model: {model_path}")

    # Create test environment
    env = CrazyFlieEnv(config=EnvConfig())

    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1
            done = terminated or truncated

            if render and episode < 3:  # Render first 3 episodes
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Check success (not crashed and reasonable performance)
        if not info.get('is_crashed', False) and episode_reward > 0:
            success_count += 1

        if verbose:
            status = "✅" if not info.get('is_crashed', False) else "❌"
            print(f"  Ep {episode+1:2d}: Reward={episode_reward:7.2f}, Steps={steps:3d} {status}")

    env.close()

    # Calculate metrics
    metrics = {
        'avg_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'avg_length': float(np.mean(episode_lengths)),
        'success_rate': success_count / num_episodes,
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards))
    }

    if verbose:
        print(f"\n📊 Results:")
        print(f"   Avg Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"   Success Rate: {metrics['success_rate']:.1%}")
        print(f"   Avg Length: {metrics['avg_length']:.1f} steps")

    return metrics['avg_reward'], metrics


def visualize_episode(model_path: str,
                     algorithm: str = "PPO",
                     max_steps: int = 1000) -> List[dict]:
    """
    Run one episode and return trajectory data for visualization.

    Args:
        model_path: Path to saved model
        algorithm: Algorithm type
        max_steps: Maximum steps per episode

    Returns:
        List of state dictionaries with positions, actions, etc.
    """

    # Load model
    if algorithm.upper() == "PPO":
        model = PPO.load(model_path)
    else:
        model = SAC.load(model_path)

    env = CrazyFlieEnv(config=EnvConfig())

    trajectory = []
    obs, info = env.reset()

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Store trajectory data
        if 'state' in obs:
            trajectory.append({
                'step': step,
                'position': obs['state'][0:3].copy(),
                'velocity': obs['state'][3:6].copy(),
                'orientation': obs['state'][6:9].copy(),
                'action': action.copy(),
                'reward': reward,
                'is_crashed': info.get('is_crashed', False)
            })

        env.render()

        obs = next_obs

        if terminated or truncated:
            break

    env.close()
    print(f"📹 Episode completed in {len(trajectory)} steps")

    return trajectory


def compare_models(model_paths: dict,
                   num_episodes: int = 10) -> dict:
    """
    Compare multiple models.

    Args:
        model_paths: Dict mapping model names to paths
        num_episodes: Episodes to test each model

    Returns:
        Dict with comparison results
    """

    print(f"🏆 Comparing {len(model_paths)} models")
    print(f"   Test episodes: {num_episodes}\n")

    results = {}

    for name, path in model_paths.items():
        print(f"Testing {name}...")

        # Detect algorithm from name or use default
        algorithm = "PPO"
        if "sac" in name.lower():
            algorithm = "SAC"

        avg_reward, metrics = test_model(
            path,
            algorithm=algorithm,
            num_episodes=num_episodes,
            render=False,
            verbose=False
        )

        results[name] = metrics
        print(f"  {name}: {avg_reward:.2f} avg reward\n")

    # Print comparison table
    print("\n📊 Comparison Results:")
    print("=" * 70)
    print(f"{'Model':<20} {'Avg Reward':>12} {'Success Rate':>14} {'Avg Length':>12}")
    print("-" * 70)

    for name, metrics in results.items():
        print(f"{name:<20} {metrics['avg_reward']:>12.2f} "
              f"{metrics['success_rate']:>13.1%} {metrics['avg_length']:>12.1f}")

    return results


def quick_test(model_path: str, algorithm: str = "PPO"):
    """
    Quick test - run 3 episodes with visualization.

    Args:
        model_path: Path to model
        algorithm: Algorithm type
    """
    print(f"🚀 Quick Test - {algorithm}")
    test_model(model_path, algorithm, num_episodes=3, render=True)
