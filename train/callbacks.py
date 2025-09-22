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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3.common.callbacks import BaseCallback
import threading
import time
from collections import deque

class LiveVisualizationCallback(BaseCallback):
    """
    Enhanced live visualization callback for drone training with proper property handling
    """
    
    def __init__(self, 
                 update_freq=100,
                 plot_freq=10,
                 max_history=1000,
                 enable_3d_plot=True,
                 enable_reward_plot=True,
                 enable_state_plot=True,
                 verbose=0):
        super().__init__(verbose)
        
        self.update_freq = update_freq
        self.plot_freq = plot_freq
        self.max_history = max_history
        self.enable_3d_plot = enable_3d_plot
        self.enable_reward_plot = enable_reward_plot
        self.enable_state_plot = enable_state_plot
        
        # Data storage
        self.episode_rewards = deque(maxlen=max_history)
        self.episode_lengths = deque(maxlen=max_history)
        self.timesteps = deque(maxlen=max_history)
        self.drone_positions = deque(maxlen=max_history)
        self.drone_velocities = deque(maxlen=max_history)
        self.heights = deque(maxlen=max_history)
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        
        # Visualization setup
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.setup_plots()
        
        # Animation and threading
        self.animation = None
        self.plot_thread = None
        self.is_plotting = False
        
        # Reference to environment (store internally, don't expose as property)
        self._env_ref = None
    
    def set_training_env(self, env):
        """Method to set the training environment reference"""
        self._env_ref = env
    
    def setup_plots(self):
        """Initialize the plotting interface"""
        if not (self.enable_3d_plot or self.enable_reward_plot or self.enable_state_plot):
            return
            
        # Calculate subplot layout
        n_plots = sum([self.enable_3d_plot, self.enable_reward_plot, self.enable_state_plot])
        
        if n_plots == 1:
            fig_size = (8, 6)
        elif n_plots == 2:
            fig_size = (12, 6)
        else:
            fig_size = (15, 10)
        
        self.fig, axes = plt.subplots(1, n_plots, figsize=fig_size)
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 3D trajectory plot
        if self.enable_3d_plot:
            # Remove the 2D axis and add 3D
            self.fig.delaxes(axes[plot_idx])
            self.axes['3d'] = self.fig.add_subplot(1, n_plots, plot_idx + 1, projection='3d')
            self.axes['3d'].set_xlabel('X Position (m)')
            self.axes['3d'].set_ylabel('Y Position (m)')
            self.axes['3d'].set_zlabel('Z Position (m)')
            self.axes['3d'].set_title('Drone 3D Trajectory')
            
            self.lines['trajectory'], = self.axes['3d'].plot([], [], [], 'b-', alpha=0.7, linewidth=2)
            self.lines['current_pos'], = self.axes['3d'].plot([], [], [], 'ro', markersize=8)
            plot_idx += 1
        
        # Reward plot
        if self.enable_reward_plot:
            self.axes['reward'] = axes[plot_idx]
            self.axes['reward'].set_xlabel('Episode')
            self.axes['reward'].set_ylabel('Total Reward')
            self.axes['reward'].set_title('Training Progress')
            self.axes['reward'].grid(True, alpha=0.3)
            
            self.lines['reward'], = self.axes['reward'].plot([], [], 'g-', linewidth=2, label='Episode Reward')
            self.lines['reward_ma'], = self.axes['reward'].plot([], [], 'r-', linewidth=2, label='Moving Average')
            self.axes['reward'].legend()
            plot_idx += 1
        
        # State monitoring plot
        if self.enable_state_plot:
            self.axes['state'] = axes[plot_idx]
            self.axes['state'].set_xlabel('Time Steps')
            self.axes['state'].set_ylabel('Values')
            self.axes['state'].set_title('Drone State Monitoring')
            self.axes['state'].grid(True, alpha=0.3)
            
            self.lines['height'], = self.axes['state'].plot([], [], 'b-', linewidth=2, label='Height (m)')
            self.lines['speed'], = self.axes['state'].plot([], [], 'r-', linewidth=2, label='Speed (m/s)')
            self.axes['state'].legend()
            plot_idx += 1
        
        plt.tight_layout()
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)
    
    def _on_training_start(self) -> None:
        """Called when training starts"""
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Start the plotting thread
        if self.fig is not None and not self.is_plotting:
            self.is_plotting = True
            self.plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
            self.plot_thread.start()
    
    def _on_step(self) -> bool:
        """Called at each step"""
        # Get current state from environment
        if hasattr(self.training_env, 'get_wrapper_attr'):
            # For vectorized environments
            try:
                env = self.training_env.get_wrapper_attr('unwrapped')[0]
            except:
                env = self.training_env
        else:
            env = self.training_env
        
        # Extract drone state if available
        if hasattr(env, 'data') and hasattr(env.data, 'qpos'):
            # MuJoCo environment
            pos = env.data.qpos[0:3].copy()
            vel = env.data.qvel[0:3].copy()
            height = pos[2]
            speed = np.linalg.norm(vel)
            
            # Store data
            self.drone_positions.append(pos)
            self.drone_velocities.append(vel)
            self.heights.append(height)
            self.timesteps.append(self.num_timesteps)
        
        # Track episode progress
        self.current_episode_length += 1
        
        # Add reward info if available
        if hasattr(self.locals, 'rewards') and self.locals['rewards'] is not None:
            reward = self.locals['rewards'][0] if isinstance(self.locals['rewards'], (list, np.ndarray)) else self.locals['rewards']
            self.current_episode_reward += reward
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        # Check if episode ended
        if hasattr(self.locals, 'dones') and self.locals['dones'] is not None:
            done = self.locals['dones'][0] if isinstance(self.locals['dones'], (list, np.ndarray)) else self.locals['dones']
            
            if done:
                # Episode finished
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.episode_count += 1
                
                # Reset episode tracking
                self.current_episode_reward = 0
                self.current_episode_length = 0
                
                if self.verbose > 0 and self.episode_count % 10 == 0:
                    avg_reward = np.mean(list(self.episode_rewards)[-10:]) if self.episode_rewards else 0
                    print(f"Episode {self.episode_count}, Avg Reward (last 10): {avg_reward:.2f}")
    
    def _plot_loop(self):
        """Main plotting loop running in separate thread"""
        while self.is_plotting:
            try:
                self.update_plots()
                time.sleep(1.0 / self.plot_freq)  # Control update frequency
            except Exception as e:
                if self.verbose > 0:
                    print(f"Plot update error: {e}")
                time.sleep(0.1)
    
    def update_plots(self):
        """Update all active plots"""
        if self.fig is None:
            return
        
        try:
            # Update 3D trajectory plot
            if self.enable_3d_plot and len(self.drone_positions) > 0:
                positions = np.array(list(self.drone_positions))
                
                # Update trajectory line
                self.lines['trajectory'].set_data_3d(positions[:, 0], positions[:, 1], positions[:, 2])
                
                # Update current position
                if len(positions) > 0:
                    current = positions[-1]
                    self.lines['current_pos'].set_data_3d([current[0]], [current[1]], [current[2]])
                
                # Auto-scale axes
                if len(positions) > 10:
                    margin = 1.0
                    self.axes['3d'].set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
                    self.axes['3d'].set_ylim(positions[:, 1].min() - margin, positions[:, 1].max() + margin)
                    self.axes['3d'].set_zlim(0, positions[:, 2].max() + margin)
            
            # Update reward plot
            if self.enable_reward_plot and len(self.episode_rewards) > 0:
                episodes = list(range(1, len(self.episode_rewards) + 1))
                rewards = list(self.episode_rewards)
                
                self.lines['reward'].set_data(episodes, rewards)
                
                # Moving average
                if len(rewards) >= 10:
                    ma_window = min(20, len(rewards))
                    moving_avg = []
                    for i in range(len(rewards)):
                        start_idx = max(0, i - ma_window + 1)
                        moving_avg.append(np.mean(rewards[start_idx:i+1]))
                    self.lines['reward_ma'].set_data(episodes, moving_avg)
                
                # Auto-scale
                self.axes['reward'].relim()
                self.axes['reward'].autoscale_view()
            
            # Update state monitoring plot
            if self.enable_state_plot and len(self.heights) > 0:
                recent_steps = min(500, len(self.heights))  # Show last 500 steps
                time_range = list(range(len(self.heights) - recent_steps, len(self.heights)))
                
                heights = list(self.heights)[-recent_steps:]
                speeds = [np.linalg.norm(vel) for vel in list(self.drone_velocities)[-recent_steps:]]
                
                self.lines['height'].set_data(time_range, heights)
                self.lines['speed'].set_data(time_range, speeds)
                
                # Auto-scale
                self.axes['state'].relim()
                self.axes['state'].autoscale_view()
            
            # Refresh the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Plot rendering error: {e}")
    
    def _on_training_end(self) -> None:
        """Called when training ends"""
        self.is_plotting = False
        if self.plot_thread and self.plot_thread.is_alive():
            self.plot_thread.join(timeout=1.0)
        
        if self.fig:
            plt.ioff()
            plt.show()  # Keep final plot open
    
    def close(self):
        """Clean up resources"""
        self.is_plotting = False
        if self.fig:
            plt.close(self.fig)



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