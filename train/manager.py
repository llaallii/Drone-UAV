# train/manager.py
"""
Main training manager that orchestrates the entire training process.
"""

import os
import json
import traceback
from datetime import datetime
from typing import Tuple, Optional, Any

try:
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("❌ Required packages not found")
    raise

from .config import TrainingConfig
from .algorithms import AlgorithmFactory
from .environments import EnvironmentFactory
from .callbacks import LiveVisualizationCallback, PerformanceMonitorCallback, SafetyMonitorCallback


class TrainingManager:
    """
    Main training manager that orchestrates the entire training process.
    
    Responsibilities:
    - Setup training directories and logging
    - Create environments and algorithms
    - Manage training lifecycle
    - Handle callbacks and monitoring
    - Save and load models
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trainer = None
        self.env = None
        self.eval_env = None
        self.model_dir = None
        self.log_dir = None
        
        # Setup directories
        self._setup_directories()
        
        # Initialize trainer
        self._initialize_trainer()
        
        print(f"🎯 Training Manager initialized for {config.algorithm}")
    
    def _setup_directories(self):
        """Setup training and logging directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = f"models/{self.config.algorithm.lower()}_drone_{timestamp}"
        self.log_dir = f"logs/{self.config.algorithm.lower()}_drone_{timestamp}"
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        print(f"📁 Model directory: {self.model_dir}")
        print(f"📁 Log directory: {self.log_dir}")
        
        # Save configuration
        self._save_config()
    
    def _save_config(self):
        """Save training configuration to file."""
        config_dict = {
            'algorithm': self.config.algorithm,
            'total_timesteps': self.config.total_timesteps,
            'num_envs': self.config.num_envs,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'enable_live_training': self.config.enable_live_training,
            'render_freq': self.config.render_freq,
            'device': self.config.device,
            'seed': self.config.seed,
            'created_at': datetime.now().isoformat()
        }
        
        config_path = os.path.join(self.model_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"💾 Configuration saved to {config_path}")
    
    def _initialize_trainer(self):
        """Initialize the appropriate trainer based on algorithm."""
        try:
            self.trainer = AlgorithmFactory.create_trainer(self.config)
            print(f"🤖 Initialized {self.config.algorithm} trainer")
        except Exception as e:
            print(f"❌ Failed to initialize trainer: {e}")
            raise
    
    def create_environments(self):
        """Create training and evaluation environments."""
        print(f"🏗️ Creating {self.config.num_envs} training environments...")
        
        try:
            # Create training environment
            self.env = EnvironmentFactory.create_training_env(
                self.config, 
                enable_live_training=self.config.enable_live_training
            )
            
            # Create evaluation environment
            self.eval_env = EnvironmentFactory.create_eval_env(self.config)
            
            print("✅ Environments created successfully")
            
        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            raise
    
    def create_callbacks(self):
        """Create training callbacks based on configuration."""
        callbacks = []
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=self.model_dir,
            log_path=self.log_dir,
            eval_freq=self.config.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=self.config.n_eval_episodes,
            verbose=1
        )
        callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=self.model_dir,
            name_prefix='checkpoint',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Performance monitoring
        performance_callback = PerformanceMonitorCallback(
            log_interval=1000,
            verbose=1
        )
        callbacks.append(performance_callback)
        
        # Safety monitoring
        safety_callback = SafetyMonitorCallback(
            crash_threshold=20,  # Stop if too many crashes
            verbose=1
        )
        callbacks.append(safety_callback)
        
        # Live training visualization
        if self.config.enable_live_training:
            live_callback = LiveVisualizationCallback(
                env=self.env,
                render_freq=self.config.render_freq,
                verbose=1
            )
            callbacks.append(live_callback)
            print("🎥 Live training visualization enabled")
        
        # Algorithm-specific callbacks
        try:
            algo_callbacks = self.trainer.get_algorithm_specific_callbacks()
            callbacks.extend(algo_callbacks)
        except Exception as e:
            print(f"⚠️ Could not add algorithm-specific callbacks: {e}")
        
        print(f"📋 Created {len(callbacks)} training callbacks")
        return callbacks
    
    def setup_logging(self):
        """Setup TensorBoard logging."""
        if self.config.log_tensorboard:
            try:
                # This will be used by the trainer
                self.tensorboard_log = self.log_dir
                print(f"📊 TensorBoard logging enabled: {self.log_dir}")
            except Exception as e:
                print(f"⚠️ TensorBoard setup failed: {e}")
    
    def train(self) -> Tuple[Optional[Any], Optional[str]]:
        """Execute the complete training process."""
        print(f"🚀 Starting {self.config.algorithm} training...")
        print(f"📊 Total timesteps: {self.config.total_timesteps:,}")
        print(f"🔧 Learning rate: {self.config.learning_rate}")
        print(f"🏗️ Environments: {self.config.num_envs}")
        print(f"💻 Device: {self.config.device}")
        
        if self.config.enable_live_training:
            print("🎥 Live visualization: ENABLED")
            print("   - Real-time environment rendering")
            print("   - Live training plots")
            print("   - Performance monitoring")
        
        print("=" * 60)
        
        try:
            # Create environments
            self.create_environments()
            
            # Setup logging
            self.setup_logging()
            
            # Create model
            print("🧠 Creating model...")
            model = self.trainer.create_model(self.env)
            
            # Set up TensorBoard logging if enabled
            if self.config.log_tensorboard:
                model.set_logger(None)  # Use default logger with our log directory
            
            # Create callbacks
            callbacks = self.create_callbacks()
            
            # Start training
            print("🎯 Training started...")
            trained_model = self.trainer.train(self.env, callbacks)
            
            # Save final model
            final_model_path = os.path.join(self.model_dir, "final_model")
            trained_model.save(final_model_path)
            
            print(f"🎉 Training completed successfully!")
            print(f"💾 Final model saved to: {final_model_path}")
            
            # Save training summary
            self._save_training_summary(success=True)
            
            return trained_model, self.model_dir
            
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
            return self._handle_interrupted_training()
            
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            traceback.print_exc()
            self._save_training_summary(success=False, error=str(e))
            return None, None
    
    def _handle_interrupted_training(self) -> Tuple[Optional[Any], Optional[str]]:
        """Handle training interruption gracefully."""
        try:
            if self.trainer.model is not None:
                interrupted_path = os.path.join(self.model_dir, "interrupted_model")
                self.trainer.model.save(interrupted_path)
                print(f"💾 Interrupted model saved to: {interrupted_path}")
                
                self._save_training_summary(success=False, interrupted=True)
                return self.trainer.model, self.model_dir
            else:
                print("⚠️ No model to save (training stopped too early)")
                return None, None
        except Exception as e:
            print(f"⚠️ Could not save interrupted model: {e}")
            return None, None
    
    def _save_training_summary(self, success: bool, error: str = None, interrupted: bool = False):
        """Save training summary to file."""
        summary = {
            'algorithm': self.config.algorithm,
            'total_timesteps': self.config.total_timesteps,
            'success': success,
            'interrupted': interrupted,
            'error': error,
            'completed_at': datetime.now().isoformat(),
            'model_dir': self.model_dir,
            'log_dir': self.log_dir
        }
        
        summary_path = os.path.join(self.model_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def cleanup(self):
        """Clean up resources after training."""
        try:
            if self.env is not None:
                self.env.close()
                print("🧹 Training environment closed")
            
            if self.eval_env is not None:
                self.eval_env.close()
                print("🧹 Evaluation environment closed")
                
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    def resume_training(self, checkpoint_path: str, additional_timesteps: int):
        """Resume training from a checkpoint."""
        print(f"🔄 Resuming training from: {checkpoint_path}")
        
        try:
            # Load the model
            model = self.trainer.model.load(checkpoint_path)
            print("✅ Checkpoint loaded successfully")
            
            # Continue training
            model.learn(
                total_timesteps=additional_timesteps,
                reset_num_timesteps=False  # Don't reset timestep counter
            )
            
            # Save resumed model
            resumed_path = os.path.join(self.model_dir, "resumed_model")
            model.save(resumed_path)
            print(f"💾 Resumed training completed: {resumed_path}")
            
            return model, self.model_dir
            
        except Exception as e:
            print(f"❌ Resume training failed: {e}")
            return None, None
    
    def get_training_status(self) -> dict:
        """Get current training status and statistics."""
        status = {
            'algorithm': self.config.algorithm,
            'model_dir': self.model_dir,
            'log_dir': self.log_dir,
            'config': {
                'total_timesteps': self.config.total_timesteps,
                'num_envs': self.config.num_envs,
                'learning_rate': self.config.learning_rate,
                'live_training': self.config.enable_live_training
            }
        }
        
        # Add model info if available
        if self.trainer and self.trainer.model:
            status['model_created'] = True
            status['model_class'] = self.trainer.model.__class__.__name__
        else:
            status['model_created'] = False
        
        # Add environment info
        if self.env:
            status['env_created'] = True
            status['num_envs_actual'] = getattr(self.env, 'num_envs', 1)
        else:
            status['env_created'] = False
        
        return status
    
    def list_checkpoints(self) -> list:
        """List available checkpoints in the model directory."""
        if not os.path.exists(self.model_dir):
            return []
        
        checkpoints = []
        for file in os.listdir(self.model_dir):
            if file.startswith('checkpoint') and file.endswith('.zip'):
                checkpoint_path = os.path.join(self.model_dir, file)
                checkpoints.append({
                    'name': file,
                    'path': checkpoint_path,
                    'size': os.path.getsize(checkpoint_path),
                    'modified': datetime.fromtimestamp(
                        os.path.getmtime(checkpoint_path)
                    ).isoformat()
                })
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x['modified'], reverse=True)
        return checkpoints
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        
        if exc_type is not None:
            print(f"⚠️ Training manager exited with exception: {exc_type.__name__}")
        
        return False  # Don't suppress exceptions