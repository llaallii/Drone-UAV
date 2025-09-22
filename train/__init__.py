# train/__init__.py
"""
Modular Training System for Autonomous Drone Navigation

This package provides a flexible, algorithm-agnostic training system
with support for multiple RL algorithms and live visualization.

Usage:
    # Interactive training
    python -m train --mode interactive
    
    # Quick training with live visualization  
    python -m train --algorithm PPO --live_training
    
    # Test a trained model
    python -m train --mode test --test_model path/to/model
"""

from .config import TrainingConfig, interactive_config_builder, create_default_configs
from .manager import TrainingManager
from .algorithms import AlgorithmFactory, BaseTrainingAlgorithm
from .callbacks import LiveVisualizationCallback, PerformanceMonitorCallback, SafetyMonitorCallback
from .environments import EnvironmentFactory, EnhancedRewardWrapper, CurriculumWrapper
from .networks import CustomCNN, SimpleVisualizationCNN, AttentionCNN
from .utils import (
    test_trained_model, benchmark_multiple_models, validate_model_file,
    create_training_report, print_system_info, get_system_info,
    estimate_training_time, print_training_estimates, run_quick_test
)

__version__ = "1.0.0"
__author__ = "Drone Navigation Team"
__description__ = "Modular RL training system for autonomous drone navigation"

# Main exports - what users typically need
__all__ = [
    # Core components
    "TrainingConfig",
    "TrainingManager", 
    "AlgorithmFactory",
    
    # Configuration helpers
    "interactive_config_builder",
    "create_default_configs",
    
    # Training utilities
    "test_trained_model",
    "run_quick_test",
    "print_system_info",
    "get_system_info",
    "validate_model_file",
    "print_training_estimates",
    
    # Advanced components (for customization)
    "BaseTrainingAlgorithm",
    "LiveVisualizationCallback",
    "PerformanceMonitorCallback",
    "SafetyMonitorCallback",
    "EnvironmentFactory",
    "EnhancedRewardWrapper",
    "CurriculumWrapper",
    "CustomCNN",
    "SimpleVisualizationCNN",
    "AttentionCNN",
    
    # Analysis and reporting
    "benchmark_multiple_models",
    "create_training_report",
    "estimate_training_time",
]

# Package metadata
__package_info__ = {
    "name": "train",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "features": [
        "Multiple RL algorithms (PPO, SAC, RAPID IRL)",
        "Live training visualization",
        "Flexible configuration system", 
        "Custom algorithm support",
        "Comprehensive monitoring and analysis",
        "Production-ready model testing"
    ],
    "algorithms": ["PPO", "SAC", "A2C", "RAPID", "CUSTOM"],
    "visualization": True,
    "live_training": True
}


def print_package_info():
    """Print information about the training package."""
    info = __package_info__
    print(f"\n🚁 {info['description']}")
    print(f"📦 Version: {info['version']}")
    print(f"👨‍💻 Author: {info['author']}")
    print(f"\n✨ Features:")
    for feature in info['features']:
        print(f"   • {feature}")
    print(f"\n🤖 Supported Algorithms: {', '.join(info['algorithms'])}")
    print(f"🎥 Live Training: {'✅' if info['live_training'] else '❌'}")
    print(f"📊 Visualization: {'✅' if info['visualization'] else '❌'}")


def quick_start_guide():
    """Print a quick start guide for new users."""
    print("\n🚀 Quick Start Guide")
    print("=" * 40)
    print("1. Interactive Mode (Recommended for beginners):")
    print("   python -m train --mode interactive")
    print("\n2. Quick Training with Live Visualization:")
    print("   python -m train --algorithm PPO --live_training")
    print("\n3. Fast Training without Visualization:")
    print("   python -m train --algorithm SAC --timesteps 500000")
    print("\n4. Test a Trained Model:")
    print("   python -m train --mode test --test_model models/*/final_model")
    print("\n5. System Check:")
    print("   python -m train --mode quick_test")
    print("\nFor more help: python -m train --help")


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    optional_deps = []
    
    # Check core dependencies
    try:
        import gymnasium
    except ImportError:
        missing_deps.append("gymnasium")
    
    try:
        import stable_baselines3
    except ImportError:
        missing_deps.append("stable-baselines3")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import mujoco
    except ImportError:
        missing_deps.append("mujoco")
    
    # Check optional dependencies
    try:
        import matplotlib
    except ImportError:
        optional_deps.append("matplotlib (for plotting)")
    
    try:
        import cv2
    except ImportError:
        optional_deps.append("opencv-python (for visualization)")
    
    try:
        import tensorboard
    except ImportError:
        optional_deps.append("tensorboard (for logging)")
    
    # Report results
    if missing_deps:
        print(f"❌ Missing required dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    if optional_deps:
        print(f"⚠️ Optional dependencies not found: {', '.join(optional_deps)}")
        print("Some features may be disabled.")
    
    print("✅ All required dependencies found!")
    return True


def get_example_configs():
    """Get example training configurations."""
    examples = {
        "beginner_ppo": TrainingConfig(
            algorithm="PPO",
            total_timesteps=100_000,
            num_envs=2,
            enable_live_training=True,
            render_freq=500
        ),
        
        "fast_sac": TrainingConfig(
            algorithm="SAC",
            total_timesteps=500_000,
            num_envs=8,
            enable_live_training=False
        ),
        
        "research_rapid": TrainingConfig(
            algorithm="RAPID",
            total_timesteps=2_000_000,
            num_envs=4,
            learning_rate=1e-4,
            enable_live_training=True
        ),
        
        "production_training": TrainingConfig(
            algorithm="PPO",
            total_timesteps=5_000_000,
            num_envs=16,
            batch_size=256,
            enable_live_training=False,
            eval_freq=100_000
        )
    }
    
    return examples


# Convenience functions for common workflows
def train_with_live_visualization(algorithm="PPO", timesteps=1_000_000):
    """Quick training with live visualization."""
    config = TrainingConfig(
        algorithm=algorithm,
        total_timesteps=timesteps,
        enable_live_training=True,
        render_freq=1000
    )
    
    with TrainingManager(config) as manager:
        return manager.train()


def quick_benchmark(algorithms=None, timesteps=100_000):
    """Quick benchmark of multiple algorithms."""
    if algorithms is None:
        algorithms = ["PPO", "SAC"]
    
    results = {}
    
    for algo in algorithms:
        print(f"\n🤖 Benchmarking {algo}...")
        config = TrainingConfig(
            algorithm=algo,
            total_timesteps=timesteps,
            num_envs=2,
            enable_live_training=False
        )
        
        try:
            with TrainingManager(config) as manager:
                model, model_dir = manager.train()
                
                if model:
                    import os
                    model_path = os.path.join(model_dir, "final_model")
                    avg_reward, success_rate = test_trained_model(
                        model_path, algo, num_episodes=5, render=False
                    )
                    results[algo] = {
                        'avg_reward': avg_reward,
                        'success_rate': success_rate
                    }
        except Exception as e:
            print(f"❌ {algo} benchmark failed: {e}")
            results[algo] = None
    
    return results


# Auto-check dependencies when package is imported
try:
    _deps_ok = check_dependencies()
    if not _deps_ok:
        print("\n⚠️ Please install missing dependencies before using the training system.")
except Exception:
    pass  # Silent fail if checks can't run


# # Print helpful info for interactive users
# if __name__ != "__main__":
#     try:
#         # Only show info if in interactive environment
#         import sys
#         if hasattr(sys, 'ps1'):  # Interactive interpreter
#             print_package_info()
#             quick_start_guide()
#     except:
#         pass