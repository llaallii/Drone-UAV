# train/callbacks.py
"""
Training callbacks for monitoring and visualization.
"""

import numpy as np
from typing import Dict, Any, Optional
import time

try:
    from stable_baselines3.common.callbacks import BaseCallback
    matplotlib_available = True
    import matplotlib.pyplot as plt
except ImportError:
    print("⚠️ Some visualization features disabled - install matplotlib")
    matplotlib_available = False
    BaseCallback = object


class LiveVisualizationCallback(BaseCallback):
    """
    Callback for live training visualization.
    
    Shows real-time training progress with:
    - Episode reward plots
    - Environment rendering
    - Training statistics
    """
    
    def __init__(self, env, render_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.render_freq = render_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.start_time = time.time()
        
        # Setup matplotlib for live plotting
        if matplotlib_available:
            plt.ion()  # Enable interactive mode
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle('Live Training Monitor', fontsize=16)
            
            # Setup subplots
            self.ax_rewards = self.axes[0, 0]
            self.ax_lengths = self.axes[0, 1] 
            self.ax_learning = self.axes[1, 0]
            self.ax_stats = self.axes[1, 1]
            
            # Configure plots
            self._setup_plots()
            plt.tight_layout()
            plt.show()
    
    def _setup_plots(self):
        """Setup the matplotlib plots."""
        if not matplotlib_available:
            return
            
        # Episode rewards plot
        self.ax_rewards.set_title('Episode Rewards')
        self.ax_rewards.set_xlabel('Episode')
        self.ax_rewards.set_ylabel('Reward')
        self.ax_rewards.grid(True, alpha=0.3)
        
        # Episode lengths plot
        self.ax_lengths.set_title('Episode Lengths')
        self.ax_lengths.set_xlabel('Episode')
        self.ax_lengths.set_ylabel('Steps')
        self.ax_lengths.grid(True, alpha=0.3)
        
        # Learning progress plot
        self.ax_learning.set_title('Learning Progress')
        self.ax_learning.set_xlabel('Training Steps')
        self.ax_learning.set_ylabel('Metrics')
        self.ax_learning.grid(True, alpha=0.3)
        
        # Statistics plot (text-based)
        self.ax_stats.set_title('Training Statistics')
        self.ax_stats.axis('off')
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Track episode progress
        if 'rewards' in self.locals:
            self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check for episode end
        if 'dones' in self.locals and self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_count += 1
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Update plots
            if matplotlib_available:
                self._update_plots()
        
        # Render environment
        if self.num_timesteps % self.render_freq == 0:
            self._render_environment()
        
        return True
    
    def _render_environment(self):
        """Render the training environment."""
        try:
            # Render the first environment
            if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
                self.env.envs[0].render()
            elif hasattr(self.env, 'render'):
                self.env.render()
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Rendering failed: {e}")
    
    def _update_plots(self):
        """Update live training plots."""
        if not matplotlib_available or len(self.episode_rewards) == 0:
            return
            
        try:
            # Update episode rewards
            self.ax_rewards.clear()
            self.ax_rewards.plot(self.episode_rewards, 'b-', alpha=0.7, label='Episode Reward')
            
            # Add moving average
            if len(self.episode_rewards) >= 10:
                window = min(10, len(self.episode_rewards))
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                episodes_avg = range(window-1, len(self.episode_rewards))
                self.ax_rewards.plot(episodes_avg, moving_avg, 'r-', linewidth=2, label='Moving Avg')
            
            self.ax_rewards.set_title(f'Episode Rewards (Episode {self.episode_count})')
            self.ax_rewards.set_xlabel('Episode')
            self.ax_rewards.set_ylabel('Reward')
            self.ax_rewards.legend()
            self.ax_rewards.grid(True, alpha=0.3)
            
            # Update episode lengths
            self.ax_lengths.clear()
            self.ax_lengths.plot(self.episode_lengths, 'g-', alpha=0.7)
            self.ax_lengths.set_title('Episode Lengths')
            self.ax_lengths.set_xlabel('Episode')
            self.ax_lengths.set_ylabel('Steps')
            self.ax_lengths.grid(True, alpha=0.3)
            
            # Update learning progress
            self.ax_learning.clear()
            if len(self.episode_rewards) > 1:
                self.ax_learning.plot(self.episode_rewards, 'purple', alpha=0.6)
                self.ax_learning.set_title(f'Training Progress (Step {self.num_timesteps})')
                self.ax_learning.set_xlabel('Episode')
                self.ax_learning.set_ylabel('Reward')
                self.ax_learning.grid(True, alpha=0.3)
            
            # Update statistics
            self._update_statistics()
            
            plt.tight_layout()
            plt.pause(0.01)  # Small pause to update plots
            
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Plot update failed: {e}")
    
    def _update_statistics(self):
        """Update training statistics display."""
        if not matplotlib_available:
            return
            
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Calculate statistics
        elapsed_time = time.time() - self.start_time
        steps_per_second = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
        
        avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0
        avg_length = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else 0
        
        # Create statistics text
        stats_text = f"""
Training Statistics:

Total Steps: {self.num_timesteps:,}
Episodes: {self.episode_count}
Elapsed Time: {elapsed_time/60:.1f} min
Steps/sec: {steps_per_second:.1f}

Recent Performance:
Avg Reward (10 ep): {avg_reward:.2f}
Avg Length (10 ep): {avg_length:.1f}

Current Episode:
Reward: {self.current_episode_reward:.2f}
Length: {self.current_episode_length}
"""
        
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    def _on_training_end(self):
        """Called when training ends."""
        if matplotlib_available:
            plt.ioff()  # Turn off interactive mode
            plt.show()  # Keep plots open
        
        # Print final statistics
        if len(self.episode_rewards) > 0:
            print(f"\n📊 Final Training Statistics:")
            print(f"   Total Episodes: {self.episode_count}")
            print(f"   Average Reward: {np.mean(self.episode_rewards):.2f}")
            print(f"   Best Reward: {np.max(self.episode_rewards):.2f}")
            print(f"   Final 10 Episodes Avg: {np.mean(self.episode_rewards[-10:]):.2f}")


class PerformanceMonitorCallback(BaseCallback):
    """Monitor training performance and efficiency."""
    
    def __init__(self, log_interval: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.start_time = time.time()
        self.episode_times = []
        self.episode_rewards = []
        self.fps_history = []
        
    def _on_step(self) -> bool:
        """Monitor performance metrics."""
        if self.num_timesteps % self.log_interval == 0:
            self._log_performance()
        
        # Track episode completion
        if 'dones' in self.locals and self.locals['dones'][0]:
            current_time = time.time()
            if hasattr(self, 'episode_start_time'):
                episode_duration = current_time - self.episode_start_time
                self.episode_times.append(episode_duration)
            
            self.episode_start_time = current_time
            
            if 'rewards' in self.locals:
                # This is approximate - actual episode reward tracking would need more logic
                pass
        
        return True
    
    def _log_performance(self):
        """Log performance statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 0:
            fps = self.num_timesteps / elapsed_time
            self.fps_history.append(fps)
            
            if self.verbose > 0:
                print(f"📈 Step {self.num_timesteps:,} | "
                      f"FPS: {fps:.1f} | "
                      f"Elapsed: {elapsed_time/60:.1f}min")


class SafetyMonitorCallback(BaseCallback):
    """Monitor for unsafe behaviors during training."""
    
    def __init__(self, crash_threshold: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.crash_threshold = crash_threshold
        self.crash_count = 0
        self.consecutive_crashes = 0
        
    def _on_step(self) -> bool:
        """Monitor for crashes and unsafe behavior."""
        # Check for crashes (this would need to be adapted based on your environment)
        if hasattr(self, 'locals') and 'infos' in self.locals:
            for info in self.locals['infos']:
                if isinstance(info, dict) and info.get('is_crashed', False):
                    self.crash_count += 1
                    self.consecutive_crashes += 1
                    
                    if self.verbose > 0:
                        print(f"⚠️ Crash detected! Total crashes: {self.crash_count}")
                    
                    # Check if too many consecutive crashes
                    if self.consecutive_crashes >= self.crash_threshold:
                        print(f"🛑 Too many consecutive crashes ({self.consecutive_crashes}). "
                              f"Consider adjusting training parameters.")
                        return False  # Stop training
                else:
                    # Reset consecutive crash counter on successful episode
                    if 'dones' in self.locals and self.locals['dones'][0]:
                        self.consecutive_crashes = 0
        
        return True


class ProgressiveRewardCallback(BaseCallback):
    """Gradually adjust reward function during training."""
    
    def __init__(self, reward_schedule: Dict[int, Dict[str, float]], verbose: int = 1):
        super().__init__(verbose)
        self.reward_schedule = reward_schedule
        self.current_weights = None
        
    def _on_step(self) -> bool:
        """Adjust reward weights based on training progress."""
        # Check if we need to update reward weights
        for timestep, weights in self.reward_schedule.items():
            if self.num_timesteps >= timestep and self.current_weights != weights:
                self.current_weights = weights
                if self.verbose > 0:
                    print(f"🎯 Updated reward weights at step {self.num_timesteps}: {weights}")
                
                # Apply weights to environment (this would need environment support)
                if hasattr(self.training_env, 'set_reward_weights'):
                    self.training_env.set_reward_weights(weights)
        
        return True


class EarlyStoppingCallback(BaseCallback):
    """Stop training early if performance criteria are met."""
    
    def __init__(self, target_reward: float, patience: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.target_reward = target_reward
        self.patience = patience
        self.best_reward = float('-inf')
        self.patience_counter = 0
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        """Check for early stopping conditions."""
        # Track episode rewards (simplified)
        if 'dones' in self.locals and self.locals['dones'][0]:
            # This is a simplified version - proper episode reward tracking would be more complex
            if hasattr(self, 'current_episode_reward'):
                self.episode_rewards.append(self.current_episode_reward)
                
                # Check recent performance
                if len(self.episode_rewards) >= 10:
                    recent_avg = np.mean(self.episode_rewards[-10:])
                    
                    if recent_avg > self.best_reward:
                        self.best_reward = recent_avg
                        self.patience_counter = 0
                        
                        if recent_avg >= self.target_reward:
                            print(f"🎯 Target reward achieved! Average: {recent_avg:.2f} >= {self.target_reward}")
                            return False  # Stop training
                    else:
                        self.patience_counter += 1
                        
                        if self.patience_counter >= self.patience:
                            print(f"⏹️ Early stopping: No improvement for {self.patience} evaluations")
                            return False
        
        return True