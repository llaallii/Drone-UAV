# train/main.py
"""
Main training entry point with command-line interface.
"""

import argparse
import traceback
import sys
from typing import Optional

from .config import TrainingConfig, interactive_config_builder, create_default_configs
from .manager import TrainingManager
from .utils import (
    test_trained_model, benchmark_multiple_models, print_system_info,
    print_training_estimates, run_quick_test, create_training_report
)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Flexible Drone Navigation Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for beginners)
  python -m train --mode interactive

  # Quick PPO training with live visualization
  python -m train --algorithm PPO --live_training --timesteps 1000000

  # SAC training without visualization (faster)
  python -m train --algorithm SAC --timesteps 500000 --num_envs 8

  # Test a trained model
  python -m train --mode test --test_model models/ppo_drone_20241201/final_model

  # Benchmark multiple algorithms
  python -m train --mode benchmark

  # Quick development test
  python -m train --mode quick_test --algorithm PPO
        """
    )
    
    # Training modes
    parser.add_argument(
        "--mode", type=str, default="interactive",
        choices=["interactive", "config", "benchmark", "test", "quick_test", "report"],
        help="Training mode (default: interactive)"
    )
    
    # Algorithm selection
    parser.add_argument(
        "--algorithm", type=str, default="PPO",
        choices=["PPO", "SAC", "DQN", "A2C", "RAPID", "CUSTOM"],
        help="RL algorithm to use (default: PPO)"
    )
    
    # Training parameters
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps (default: 500000)"
    )
    parser.add_argument(
        "--num_envs", type=int, default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size (default: 64)"
    )
    
    # Visualization
    parser.add_argument(
        "--live_training", action="store_true",
        help="Enable live training visualization"
    )
    parser.add_argument(
        "--render_freq", type=int, default=1000,
        help="Rendering frequency during training (default: 1000)"
    )
    
    # Evaluation and saving
    parser.add_argument(
        "--eval_freq", type=int, default=25000,
        help="Evaluation frequency (default: 25000)"
    )
    parser.add_argument(
        "--save_freq", type=int, default=50000,
        help="Model save frequency (default: 50000)"
    )
    
    # Testing
    parser.add_argument(
        "--test_model", type=str, default=None,
        help="Path to model for testing"
    )
    parser.add_argument(
        "--test_episodes", type=int, default=10,
        help="Number of test episodes (default: 10)"
    )
    
    # System
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    
    # Utility options
    parser.add_argument(
        "--system_info", action="store_true",
        help="Print system information and exit"
    )
    parser.add_argument(
        "--list_algorithms", action="store_true",
        help="List available algorithms and exit"
    )
    
    return parser


def handle_interactive_mode() -> Optional[TrainingConfig]:
    """Handle interactive configuration mode."""
    try:
        config = interactive_config_builder()
        return config
    except KeyboardInterrupt:
        print("\n👋 Interactive mode cancelled by user")
        return None
    except Exception as e:
        print(f"❌ Interactive mode failed: {e}")
        return None


def handle_config_mode(args) -> TrainingConfig:
    """Handle direct configuration mode."""
    config = TrainingConfig(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        enable_live_training=args.live_training,
        render_freq=args.render_freq,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        device=args.device,
        seed=args.seed
    )
    
    print(f"🔧 Configuration created:")
    print(f"   Algorithm: {config.algorithm}")
    print(f"   Timesteps: {config.total_timesteps:,}")
    print(f"   Environments: {config.num_envs}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Live Training: {'✅' if config.enable_live_training else '❌'}")
    
    return config


def handle_benchmark_mode():
    """Handle benchmark mode."""
    print("🏁 Algorithm Benchmark Mode")
    print("=" * 40)
    
    algorithms_to_test = ['PPO', 'SAC']
    timesteps_per_algo = 100_000  # Shorter for benchmarking
    
    results = {}
    
    for algorithm in algorithms_to_test:
        print(f"\n🤖 Benchmarking {algorithm}...")
        
        config = TrainingConfig(
            algorithm=algorithm,
            total_timesteps=timesteps_per_algo,
            num_envs=2,  # Fewer environments for faster benchmarking
            enable_live_training=False
        )
        
        try:
            with TrainingManager(config) as manager:
                model, model_dir = manager.train()
                
                if model is not None:
                    # Test the trained model
                    import os
                    model_path = os.path.join(model_dir, "final_model")
                    avg_reward, success_rate = test_trained_model(
                        model_path, algorithm, num_episodes=5, render=False
                    )
                    
                    results[algorithm] = {
                        'avg_reward': avg_reward,
                        'success_rate': success_rate,
                        'model_path': model_path
                    }
        
        except Exception as e:
            print(f"❌ {algorithm} benchmark failed: {e}")
            results[algorithm] = None
    
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


def handle_test_mode(args):
    """Handle model testing mode."""
    if args.test_model is None:
        print("❌ --test_model path required for test mode")
        return
    
    print(f"🧪 Testing model: {args.test_model}")
    
    avg_reward, success_rate = test_trained_model(
        model_path=args.test_model,
        algorithm=args.algorithm,
        num_episodes=args.test_episodes,
        render=True
    )
    
    if avg_reward is not None:
        print(f"\n🎯 Test completed successfully!")
        return {'avg_reward': avg_reward, 'success_rate': success_rate}
    else:
        print(f"❌ Test failed")
        return None


def handle_quick_test_mode(args):
    """Handle quick test mode."""
    print(f"🧪 Quick Test Mode - {args.algorithm}")
    print("   This will run a short training to verify everything works")
    
    success = run_quick_test(args.algorithm)
    
    if success:
        print("✅ Quick test completed successfully!")
        print("   Your training system is working correctly")
    else:
        print("❌ Quick test failed")
        print("   Check your environment setup and dependencies")
    
    return success


def handle_report_mode(args):
    """Handle training report generation."""
    if args.test_model is None:
        print("❌ --test_model path required for report mode")
        print("   Provide path to model directory (not the .zip file)")
        return
    
    # Extract model directory from model path
    import os
    if args.test_model.endswith('.zip'):
        model_dir = os.path.dirname(args.test_model)
    else:
        model_dir = args.test_model
    
    print(f"📝 Generating training report for: {model_dir}")
    
    report = create_training_report(model_dir, test_episodes=args.test_episodes)
    print("\n" + report)
    
    return report


def ask_live_training_preference(config: TrainingConfig) -> TrainingConfig:
    """Ask user about live training preference if not specified."""
    if not config.enable_live_training:
        try:
            live_prompt = input("\n🎥 Would you like to see live training visualization? [y/N]: ")
            if live_prompt.strip().lower() in ['y', 'yes', '1', 'true']:
                config.enable_live_training = True
                print("✅ Live training visualization enabled!")
                print("   You will see:")
                print("   - Real-time environment rendering")
                print("   - Live reward plots")
                print("   - Training progress graphs")
                print("   - Performance statistics")
        except (KeyboardInterrupt, EOFError):
            print("\n   Continuing without live training...")
    
    return config


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle utility options
    if args.system_info:
        print_system_info()
        return
    
    if args.list_algorithms:
        from .algorithms import AlgorithmFactory
        algorithms = AlgorithmFactory.list_algorithms()
        print("🤖 Available algorithms:")
        for algo in algorithms:
            print(f"   {algo}")
        return
    
    print("🚁 Autonomous Drone Navigation Training System")
    print("=" * 50)
    
    try:
        # Handle different modes
        if args.mode == "interactive":
            config = handle_interactive_mode()
            if config is None:
                return
        
        elif args.mode == "benchmark":
            handle_benchmark_mode()
            return
        
        elif args.mode == "test":
            handle_test_mode(args)
            return
        
        elif args.mode == "quick_test":
            handle_quick_test_mode(args)
            return
        
        elif args.mode == "report":
            handle_report_mode(args)
            return
        
        else:  # config mode
            config = handle_config_mode(args)
        
        # Ask about live training if not specified in interactive mode
        if args.mode != "interactive":
            config = ask_live_training_preference(config)
        
        # Print training estimates
        print_training_estimates(config)
        
        # Confirm before starting long training
        if config.total_timesteps > 100_000:
            try:
                confirm = input(f"\n🚀 Start training {config.algorithm} for {config.total_timesteps:,} steps? [Y/n]: ")
                if confirm.strip().lower() in ['n', 'no']:
                    print("👋 Training cancelled by user")
                    return
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Training cancelled by user")
                return
        
        # Execute training
        print(f"\n🚀 Starting training with {config.algorithm}...")
        if config.enable_live_training:
            print("🎥 Live visualization active - you'll see the drone learning in real-time!")
        
        with TrainingManager(config) as manager:
            result = manager.train()
            
            if result[0] is not None:
                model, model_dir = result
                print(f"\n🧪 Testing trained {config.algorithm} model...")
                
                # Test the trained model
                import os
                model_path = os.path.join(model_dir, "final_model")
                test_trained_model(model_path, config.algorithm, num_episodes=5)
                
                # Generate training report
                print(f"\n📝 Generating training report...")
                create_training_report(model_dir)
                
            else:
                print("❌ Training failed - no model to test")
    
    except KeyboardInterrupt:
        print("\n👋 Training system interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()