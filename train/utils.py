# train/utils.py
"""
Utility functions for training and testing.
"""

import os
import numpy as np
import traceback
from typing import Tuple, Optional, Dict, Any, List

try:
    from stable_baselines3 import PPO, SAC, DQN, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("❌ Stable Baselines3 not found")
    raise

from .environments import EnvironmentFactory
from .config import TrainingConfig


def test_trained_model(model_path: str, algorithm: str = "PPO", 
                      num_episodes: int = 10, render: bool = True,
                      env_config=None) -> Tuple[Optional[float], Optional[float]]:
    """
    Test a trained model and return performance metrics.
    
    Args:
        model_path: Path to the saved model
        algorithm: Algorithm used to train the model
        num_episodes: Number of test episodes
        render: Whether to render during testing
        env_config: Optional environment configuration
        
    Returns:
        Tuple of (average_reward, success_rate)
    """
    print(f"🧪 Testing {algorithm} model: {model_path}")
    
    try:
        # Load model based on algorithm
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
        test_config = TrainingConfig(
            algorithm=algorithm,
            num_envs=1,
            env_config=env_config
        )
        
        env = EnvironmentFactory.create_eval_env(test_config)
        
        # Run test episodes
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        crash_count = 0
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while episode_length < 1000:  # Max steps per episode
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                episode_length += 1
                
                # Render every few episodes
                if render and episode < 3:
                    try:
                        if hasattr(env, 'envs') and len(env.envs) > 0:
                            env.envs[0].render()
                    except:
                        pass
                
                if done[0]:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Analyze episode outcome
            episode_info = info[0] if isinstance(info, list) else info
            
            # Check success criteria
            goal_distance = episode_info.get('distance_to_goal', float('inf'))
            is_crashed = episode_info.get('is_crashed', False)
            
            if goal_distance < 1.0 and not is_crashed:
                success_count += 1
                status = "SUCCESS ✅"
            elif is_crashed:
                crash_count += 1
                status = "CRASHED ❌"
            else:
                status = "INCOMPLETE ⏸️"
            
            print(f"Episode {episode + 1:2d}: Reward={episode_reward:7.2f}, "
                  f"Length={episode_length:3d}, {status}")
        
        # Calculate statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        success_rate = success_count / num_episodes
        crash_rate = crash_count / num_episodes
        
        # Print summary
        print(f"\n📊 Test Results Summary:")
        print(f"   Episodes: {num_episodes}")
        print(f"   Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"   Average Length: {avg_length:.1f} steps")
        print(f"   Success Rate: {success_rate:.1%} ({success_count}/{num_episodes})")
        print(f"   Crash Rate: {crash_rate:.1%} ({crash_count}/{num_episodes})")
        print(f"   Best Episode: {np.max(episode_rewards):.2f}")
        print(f"   Worst Episode: {np.min(episode_rewards):.2f}")
        
        env.close()
        return avg_reward, success_rate
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        traceback.print_exc()
        return None, None


def benchmark_multiple_models(model_paths: Dict[str, str], num_episodes: int = 10,
                             env_config=None) -> Dict[str, Dict[str, float]]:
    """
    Benchmark multiple models and compare their performance.
    
    Args:
        model_paths: Dictionary mapping model names to paths
        num_episodes: Number of episodes to test each model
        env_config: Environment configuration
        
    Returns:
        Dictionary with performance metrics for each model
    """
    print(f"🏁 Benchmarking {len(model_paths)} models with {num_episodes} episodes each")
    
    results = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\n🤖 Testing {model_name}...")
        
        # Extract algorithm from model name or path
        algorithm = "PPO"  # Default
        for algo in ["PPO", "SAC", "DQN", "A2C"]:
            if algo.lower() in model_name.lower() or algo.lower() in model_path.lower():
                algorithm = algo
                break
        
        avg_reward, success_rate = test_trained_model(
            model_path=model_path,
            algorithm=algorithm,
            num_episodes=num_episodes,
            render=False,  # No rendering for benchmarking
            env_config=env_config
        )
        
        if avg_reward is not None:
            results[model_name] = {
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'algorithm': algorithm,
                'model_path': model_path
            }
        else:
            results[model_name] = None
    
    # Print comparison table
    print(f"\n📊 Benchmark Results:")
    print("=" * 80)
    print(f"{'Model Name':<20} {'Algorithm':<10} {'Avg Reward':<12} {'Success Rate':<12}")
    print("-" * 80)
    
    for model_name, result in results.items():
        if result is not None:
            print(f"{model_name:<20} {result['algorithm']:<10} "
                  f"{result['avg_reward']:<12.2f} {result['success_rate']:<12.1%}")
        else:
            print(f"{model_name:<20} {'FAILED':<10} {'N/A':<12} {'N/A':<12}")
    
    return results


def find_best_model(model_directory: str, metric: str = 'reward') -> Optional[str]:
    """
    Find the best model in a directory based on a metric.
    
    Args:
        model_directory: Directory containing model files
        metric: Metric to optimize ('reward' or 'success_rate')
        
    Returns:
        Path to the best model file
    """
    if not os.path.exists(model_directory):
        print(f"❌ Directory not found: {model_directory}")
        return None
    
    model_files = []
    for file in os.listdir(model_directory):
        if file.endswith('.zip'):
            model_path = os.path.join(model_directory, file)
            model_files.append((file, model_path))
    
    if not model_files:
        print(f"❌ No model files found in {model_directory}")
        return None
    
    print(f"🔍 Found {len(model_files)} models, testing to find best by {metric}...")
    
    best_model = None
    best_score = float('-inf')
    
    for model_name, model_path in model_files:
        print(f"   Testing {model_name}...")
        
        avg_reward, success_rate = test_trained_model(
            model_path=model_path,
            num_episodes=5,  # Quick test
            render=False
        )
        
        if avg_reward is not None:
            score = avg_reward if metric == 'reward' else success_rate
            if score > best_score:
                best_score = score
                best_model = model_path
    
    if best_model:
        print(f"🏆 Best model: {os.path.basename(best_model)} (score: {best_score:.3f})")
    
    return best_model


def analyze_training_logs(log_directory: str) -> Dict[str, Any]:
    """
    Analyze training logs and extract useful statistics.
    
    Args:
        log_directory: Directory containing training logs
        
    Returns:
        Dictionary with training analysis
    """
    analysis = {
        'log_directory': log_directory,
        'files_found': [],
        'training_progress': {},
        'performance_metrics': {}
    }
    
    if not os.path.exists(log_directory):
        print(f"❌ Log directory not found: {log_directory}")
        return analysis
    
    # Find log files
    for file in os.listdir(log_directory):
        if file.endswith('.json'):
            analysis['files_found'].append(file)
    
    # Try to read training config
    config_path = os.path.join(log_directory, 'training_config.json')
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, 'r') as f:
                analysis['training_config'] = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not read training config: {e}")
    
    # Try to read training summary
    summary_path = os.path.join(log_directory, 'training_summary.json')
    if os.path.exists(summary_path):
        try:
            import json
            with open(summary_path, 'r') as f:
                analysis['training_summary'] = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not read training summary: {e}")
    
    print(f"📊 Training log analysis for: {log_directory}")
    print(f"   Files found: {len(analysis['files_found'])}")
    
    if 'training_config' in analysis:
        config = analysis['training_config']
        print(f"   Algorithm: {config.get('algorithm', 'Unknown')}")
        print(f"   Total timesteps: {config.get('total_timesteps', 'Unknown'):,}")
        print(f"   Learning rate: {config.get('learning_rate', 'Unknown')}")
    
    if 'training_summary' in analysis:
        summary = analysis['training_summary']
        print(f"   Training success: {summary.get('success', 'Unknown')}")
        if 'completed_at' in summary:
            print(f"   Completed: {summary['completed_at']}")
    
    return analysis


def create_training_report(model_directory: str, test_episodes: int = 20) -> str:
    """
    Create a comprehensive training report.
    
    Args:
        model_directory: Directory containing model and logs
        test_episodes: Number of episodes for final evaluation
        
    Returns:
        Report as formatted string
    """
    report_lines = []
    report_lines.append("# 🚁 Drone Training Report")
    report_lines.append("=" * 50)
    
    # Basic info
    report_lines.append(f"Model Directory: {model_directory}")
    report_lines.append(f"Generated: {np.datetime64('now')}")
    report_lines.append("")
    
    # Analyze logs
    log_dir = model_directory.replace('models/', 'logs/')
    if os.path.exists(log_dir):
        analysis = analyze_training_logs(log_dir)
        
        if 'training_config' in analysis:
            config = analysis['training_config']
            report_lines.append("## Training Configuration")
            report_lines.append(f"- Algorithm: {config.get('algorithm', 'Unknown')}")
            report_lines.append(f"- Total timesteps: {config.get('total_timesteps', 'Unknown'):,}")
            report_lines.append(f"- Learning rate: {config.get('learning_rate', 'Unknown')}")
            report_lines.append(f"- Batch size: {config.get('batch_size', 'Unknown')}")
            report_lines.append(f"- Number of environments: {config.get('num_envs', 'Unknown')}")
            report_lines.append("")
    
    # Test final model
    final_model_path = os.path.join(model_directory, 'final_model.zip')
    if os.path.exists(final_model_path):
        report_lines.append("## Model Performance")
        
        avg_reward, success_rate = test_trained_model(
            model_path=final_model_path,
            num_episodes=test_episodes,
            render=False
        )
        
        if avg_reward is not None:
            report_lines.append(f"- Average reward: {avg_reward:.2f}")
            report_lines.append(f"- Success rate: {success_rate:.1%}")
            report_lines.append(f"- Test episodes: {test_episodes}")
        else:
            report_lines.append("- Model testing failed")
        report_lines.append("")
    
    # List available checkpoints
    checkpoints = []
    for file in os.listdir(model_directory):
        if file.startswith('checkpoint') and file.endswith('.zip'):
            checkpoints.append(file)
    
    if checkpoints:
        report_lines.append("## Available Checkpoints")
        for checkpoint in sorted(checkpoints):
            report_lines.append(f"- {checkpoint}")
        report_lines.append("")
    
    # Training recommendations
    report_lines.append("## Recommendations")
    if avg_reward is not None:
        if success_rate < 0.3:
            report_lines.append("- ⚠️ Low success rate - consider longer training or hyperparameter tuning")
        elif success_rate > 0.8:
            report_lines.append("- ✅ High success rate - model is performing well")
        else:
            report_lines.append("- 📈 Moderate success rate - some improvement possible")
        
        if avg_reward < -50:
            report_lines.append("- ⚠️ Low average reward - check reward function and training stability")
        elif avg_reward > 50:
            report_lines.append("- ✅ Good average reward - model has learned effective policies")
    
    report = "\n".join(report_lines)
    
    # Save report to file
    report_path = os.path.join(model_directory, 'training_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📝 Training report saved to: {report_path}")
    return report


def cleanup_old_models(models_directory: str = "models", keep_latest: int = 5):
    """
    Clean up old model directories, keeping only the most recent ones.
    
    Args:
        models_directory: Directory containing model subdirectories
        keep_latest: Number of latest models to keep
    """
    if not os.path.exists(models_directory):
        print(f"❌ Models directory not found: {models_directory}")
        return
    
    # Find all model directories
    model_dirs = []
    for item in os.listdir(models_directory):
        item_path = os.path.join(models_directory, item)
        if os.path.isdir(item_path):
            # Get modification time
            mtime = os.path.getmtime(item_path)
            model_dirs.append((item, item_path, mtime))
    
    # Sort by modification time (newest first)
    model_dirs.sort(key=lambda x: x[2], reverse=True)
    
    # Remove old directories
    if len(model_dirs) > keep_latest:
        dirs_to_remove = model_dirs[keep_latest:]
        
        print(f"🧹 Cleaning up {len(dirs_to_remove)} old model directories...")
        
        for dir_name, dir_path, _ in dirs_to_remove:
            try:
                import shutil
                shutil.rmtree(dir_path)
                print(f"   Removed: {dir_name}")
            except Exception as e:
                print(f"   ⚠️ Could not remove {dir_name}: {e}")
    
    print(f"✅ Cleanup complete. Kept {min(len(model_dirs), keep_latest)} most recent models.")


def validate_model_file(model_path: str) -> bool:
    """
    Validate that a model file can be loaded correctly.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if model is valid, False otherwise
    """
    try:
        # Try to detect algorithm from path
        algorithm = "PPO"
        for algo in ["PPO", "SAC", "DQN", "A2C"]:
            if algo.lower() in model_path.lower():
                algorithm = algo
                break
        
        # Try to load the model
        algorithm_map = {
            'PPO': PPO,
            'SAC': SAC,
            'DQN': DQN,
            'A2C': A2C
        }
        
        model_class = algorithm_map.get(algorithm, PPO)
        model = model_class.load(model_path)
        
        print(f"✅ Model validation passed: {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {model_path}")
        print(f"   Error: {e}")
        return False


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging and optimization."""
    import platform
    import psutil
    import torch
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
        info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    
    return info


def print_system_info():
    """Print system information for debugging."""
    info = get_system_info()
    
    print("💻 System Information:")
    print(f"   Platform: {info['platform']}")
    print(f"   Python: {info['python_version']}")
    print(f"   CPU Cores: {info['cpu_count']}")
    print(f"   RAM: {info['memory_gb']} GB")
    print(f"   PyTorch: {info['torch_version']}")
    
    if info['cuda_available']:
        print(f"   CUDA: {info['cuda_version']}")
        print(f"   GPU: {info['gpu_name']} ({info['gpu_memory_gb']} GB)")
        print(f"   GPU Count: {info['gpu_count']}")
    else:
        print("   CUDA: Not available")


def estimate_training_time(config: TrainingConfig) -> Dict[str, float]:
    """
    Estimate training time based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary with time estimates
    """
    # Base estimates (steps per second) for different algorithms
    base_fps = {
        'PPO': 2000,
        'SAC': 1500,
        'DQN': 1800,
        'A2C': 2500,
        'RAPID': 1000
    }
    
    fps = base_fps.get(config.algorithm, 1500)
    
    # Adjust for number of environments
    fps *= config.num_envs
    
    # Adjust for device
    if config.device == 'cuda':
        fps *= 2.0  # GPU speedup
    
    # Adjust for live training (slower due to rendering)
    if config.enable_live_training:
        fps *= 0.7
    
    # Calculate estimates
    total_seconds = config.total_timesteps / fps
    
    estimates = {
        'steps_per_second': fps,
        'total_seconds': total_seconds,
        'total_minutes': total_seconds / 60,
        'total_hours': total_seconds / 3600,
        'estimated_completion': f"{total_seconds/3600:.1f} hours"
    }
    
    return estimates


def print_training_estimates(config: TrainingConfig):
    """Print training time estimates."""
    estimates = estimate_training_time(config)
    
    print("⏱️ Training Time Estimates:")
    print(f"   Steps per second: ~{estimates['steps_per_second']:.0f}")
    print(f"   Total time: {estimates['estimated_completion']}")
    
    if estimates['total_hours'] > 24:
        days = estimates['total_hours'] / 24
        print(f"   (approximately {days:.1f} days)")
    elif estimates['total_hours'] > 1:
        print(f"   (approximately {estimates['total_hours']:.1f} hours)")
    else:
        print(f"   (approximately {estimates['total_minutes']:.0f} minutes)")


def create_quick_test_config(algorithm: str = "PPO") -> TrainingConfig:
    """Create a quick test configuration for development."""
    return TrainingConfig(
        algorithm=algorithm,
        total_timesteps=10_000,  # Very short for testing
        num_envs=2,
        learning_rate=1e-3,  # Higher learning rate for faster learning
        enable_live_training=True,
        render_freq=100,  # Frequent rendering for testing
        eval_freq=2_000,
        save_freq=5_000
    )


def run_quick_test(algorithm: str = "PPO"):
    """Run a quick test to verify everything works."""
    print(f"🧪 Running quick test with {algorithm}...")
    
    from .manager import TrainingManager
    
    config = create_quick_test_config(algorithm)
    
    try:
        with TrainingManager(config) as manager:
            model, model_dir = manager.train()
            
            if model is not None:
                print("✅ Quick test completed successfully!")
                
                # Quick evaluation
                model_path = os.path.join(model_dir, "final_model")
                avg_reward, success_rate = test_trained_model(
                    model_path, algorithm, num_episodes=3, render=False
                )
                
                if avg_reward is not None:
                    print(f"   Test reward: {avg_reward:.2f}")
                    print(f"   Success rate: {success_rate:.1%}")
                
                return True
            else:
                print("❌ Quick test failed - no model created")
                return False
                
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


# Utility functions for data analysis
def load_training_metrics(log_directory: str) -> Optional[Dict[str, Any]]:
    """Load training metrics from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        event_acc = EventAccumulator(log_directory)
        event_acc.Reload()
        
        metrics = {}
        
        # Get scalar metrics
        for tag in event_acc.Tags()['scalars']:
            scalar_events = event_acc.Scalars(tag)
            metrics[tag] = {
                'steps': [event.step for event in scalar_events],
                'values': [event.value for event in scalar_events]
            }
        
        return metrics
        
    except ImportError:
        print("⚠️ TensorBoard not available for metrics loading")
        return None
    except Exception as e:
        print(f"⚠️ Could not load training metrics: {e}")
        return None


def plot_training_curves(log_directory: str, save_path: str = None):
    """Plot training curves from logs."""
    try:
        import matplotlib.pyplot as plt
        
        metrics = load_training_metrics(log_directory)
        if metrics is None:
            print("❌ Could not load metrics for plotting")
            return
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Plot episode rewards
        if 'rollout/ep_rew_mean' in metrics:
            ax = axes[0, 0]
            data = metrics['rollout/ep_rew_mean']
            ax.plot(data['steps'], data['values'])
            ax.set_title('Episode Reward')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Reward')
            ax.grid(True)
        
        # Plot episode length
        if 'rollout/ep_len_mean' in metrics:
            ax = axes[0, 1]
            data = metrics['rollout/ep_len_mean']
            ax.plot(data['steps'], data['values'])
            ax.set_title('Episode Length')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Length')
            ax.grid(True)
        
        # Plot learning rate
        if 'train/learning_rate' in metrics:
            ax = axes[1, 0]
            data = metrics['train/learning_rate']
            ax.plot(data['steps'], data['values'])
            ax.set_title('Learning Rate')
            ax.set_xlabel('Steps')
            ax.set_ylabel('LR')
            ax.grid(True)
        
        # Plot loss
        loss_metrics = [k for k in metrics.keys() if 'loss' in k.lower()]
        if loss_metrics:
            ax = axes[1, 1]
            for metric in loss_metrics[:3]:  # Plot up to 3 loss metrics
                data = metrics[metric]
                ax.plot(data['steps'], data['values'], label=metric.split('/')[-1])
            ax.set_title('Training Loss')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training curves saved to: {save_path}")
        else:
            plt.show()
        
        return fig
        
    except ImportError:
        print("⚠️ Matplotlib not available for plotting")
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")


def export_model_for_deployment(model_path: str, output_path: str, 
                               format: str = 'onnx') -> bool:
    """
    Export model for deployment in different formats.
    
    Args:
        model_path: Path to trained model
        output_path: Output path for exported model
        format: Export format ('onnx', 'torchscript', etc.)
        
    Returns:
        True if export successful
    """
    print(f"📦 Exporting model to {format.upper()} format...")
    
    try:
        # This would require additional implementation based on deployment needs
        # For now, just copy the model
        import shutil
        shutil.copy2(model_path, output_path)
        
        print(f"✅ Model exported to: {output_path}")
        print("⚠️ Note: Advanced export formats not yet implemented")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False