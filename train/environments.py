# train/environments.py
"""
Environment factory and wrapper classes.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional, Callable
from collections import deque

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
except ImportError:
    print("❌ Stable Baselines3 not found")
    raise

try:
    from crazy_flie_env import CrazyFlieEnv
    from crazy_flie_env.utils.config import EnvConfig
except ImportError:
    print("⚠️ CrazyFlieEnv not found - using mock environment")
    CrazyFlieEnv = None
    EnvConfig = None


class EnhancedRewardWrapper(gym.Wrapper):
    """Enhanced reward wrapper with configurable reward components."""
    
    def __init__(self, env, reward_config: Dict[str, float] = None):
        super().__init__(env)
        self.reward_config = reward_config or {
            'distance_progress': 10.0,
            'altitude_error': 0.5,
            'stability': 0.1,
            'efficiency': 0.01,
            'collision': -100.0,
            'goal_reached': 100.0,
            'time_penalty': -0.01
        }
        
        # Episode tracking
        self.prev_distance_to_goal = None
        self.goal_position = None
        self.episode_step = 0
        self.max_episode_steps = 1000
        
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
        if isinstance(obs, dict) and 'state' in obs:
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
        info['episode_step'] = self.episode_step
        
        # Check episode length
        if self.episode_step >= self.max_episode_steps:
            truncated = True
        
        return obs, enhanced_reward, terminated, truncated, info
    
    def _calculate_enhanced_reward(self, obs, action, terminated, info):
        """Calculate enhanced reward with multiple components."""
        if not isinstance(obs, dict) or 'state' not in obs:
            return 0.0
        
        state = obs['state']
        current_position = state[0:3]
        velocity = state[3:6]
        angular_velocity = state[9:12]
        
        reward_components = {}
        
        # Distance progress reward
        if self.prev_distance_to_goal is not None:
            current_distance = np.linalg.norm(self.goal_position - current_position)
            progress = self.prev_distance_to_goal - current_distance
            reward_components['distance_progress'] = progress * self.reward_config['distance_progress']
            self.prev_distance_to_goal = current_distance
            
            # Goal reached bonus
            if current_distance < 0.5:
                reward_components['goal_reached'] = self.reward_config['goal_reached']
                terminated = True
        else:
            reward_components['distance_progress'] = 0.0
            reward_components['goal_reached'] = 0.0
        
        # Altitude error penalty
        target_altitude = 1.5
        altitude_error = abs(current_position[2] - target_altitude)
        reward_components['altitude_error'] = -altitude_error * self.reward_config['altitude_error']
        
        # Stability reward
        angular_speed = np.linalg.norm(angular_velocity)
        reward_components['stability'] = -angular_speed * self.reward_config['stability']
        
        # Efficiency penalty
        action_magnitude = np.linalg.norm(action)
        reward_components['efficiency'] = -action_magnitude * self.reward_config['efficiency']
        
        # Collision penalty
        if info.get('is_crashed', False):
            reward_components['collision'] = self.reward_config['collision']
        else:
            reward_components['collision'] = 0.0
        
        # Time penalty
        reward_components['time_penalty'] = self.reward_config['time_penalty']
        
        total_reward = sum(reward_components.values())
        
        # Store breakdown for analysis
        info['reward_breakdown'] = reward_components
        
        return total_reward
    
    def set_reward_weights(self, new_weights: Dict[str, float]):
        """Update reward component weights."""
        self.reward_config.update(new_weights)
        print(f"🎯 Updated reward weights: {new_weights}")


class CurriculumWrapper(gym.Wrapper):
    """Curriculum learning wrapper to gradually increase difficulty."""
    
    def __init__(self, env, curriculum_config: Dict[str, Any] = None):
        super().__init__(env)
        self.episode_count = 0
        self.success_rate = deque(maxlen=100)
        self.current_level = 0
        
        # Default curriculum levels
        self.curriculum_levels = curriculum_config or [
            {'name': 'basic', 'goal_distance': 5.0, 'max_speed': 3.0, 'success_threshold': 0.7},
            {'name': 'intermediate', 'goal_distance': 10.0, 'max_speed': 5.0, 'success_threshold': 0.6},
            {'name': 'advanced', 'goal_distance': 15.0, 'max_speed': 7.0, 'success_threshold': 0.5},
            {'name': 'expert', 'goal_distance': 20.0, 'max_speed': 8.0, 'success_threshold': 0.4}
        ]
        
        print(f"📚 Curriculum learning initialized with {len(self.curriculum_levels)} levels")
        
    def reset(self, **kwargs):
        """Reset with curriculum level consideration."""
        self.episode_count += 1
        
        # Check for level advancement
        if len(self.success_rate) >= 50:
            current_success = np.mean(self.success_rate)
            level_config = self.curriculum_levels[self.current_level]
            
            if current_success > level_config['success_threshold']:
                if self.current_level < len(self.curriculum_levels) - 1:
                    self.current_level += 1
                    self.success_rate.clear()
                    print(f"🎓 Advanced to curriculum level {self.current_level}: "
                          f"{self.curriculum_levels[self.current_level]['name']}")
        
        obs, info = self.env.reset(**kwargs)
        info['curriculum_level'] = self.current_level
        info['curriculum_config'] = self.curriculum_levels[self.current_level]
        
        return obs, info
    
    def step(self, action):
        """Step with curriculum tracking."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track success for curriculum progression
        if terminated or truncated:
            # Define success criteria (customize based on your task)
            distance_to_goal = info.get('distance_to_goal', float('inf'))
            success = distance_to_goal < 1.0 and not info.get('is_crashed', False)
            self.success_rate.append(float(success))
        
        return obs, reward, terminated, truncated, info


class ObservationNormalizationWrapper(gym.ObservationWrapper):
    """Normalize observations for stable training."""
    
    def __init__(self, env, normalize_images: bool = True, normalize_states: bool = True):
        super().__init__(env)
        self.normalize_images = normalize_images
        self.normalize_states = normalize_states
        
        # State normalization statistics
        self.state_mean = None
        self.state_std = None
        self.update_stats = True
        self.stats_samples = 0
        self.max_stats_samples = 10000
        
    def observation(self, obs):
        """Normalize observations."""
        if not isinstance(obs, dict):
            return obs
        
        normalized_obs = obs.copy()
        
        # Normalize images
        if self.normalize_images and 'image' in obs:
            # Images are typically already normalized in the network
            pass
        
        # Normalize state vector
        if self.normalize_states and 'state' in obs:
            state = obs['state'].astype(np.float32)
            
            # Update statistics
            if self.update_stats and self.stats_samples < self.max_stats_samples:
                if self.state_mean is None:
                    self.state_mean = state.copy()
                    self.state_std = np.ones_like(state)
                else:
                    # Running average
                    alpha = 1.0 / (self.stats_samples + 1)
                    self.state_mean = (1 - alpha) * self.state_mean + alpha * state
                    self.state_std = (1 - alpha) * self.state_std + alpha * np.abs(state - self.state_mean)
                
                self.stats_samples += 1
                
                if self.stats_samples >= self.max_stats_samples:
                    self.update_stats = False
                    print(f"📊 State normalization statistics computed from {self.stats_samples} samples")
            
            # Apply normalization
            if self.state_mean is not None:
                normalized_state = (state - self.state_mean) / (self.state_std + 1e-8)
                normalized_obs['state'] = normalized_state
        
        return normalized_obs


class ActionSmoothingWrapper(gym.ActionWrapper):
    """Smooth actions to reduce jerkiness."""
    
    def __init__(self, env, smoothing_factor: float = 0.8):
        super().__init__(env)
        self.smoothing_factor = smoothing_factor
        self.prev_action = None
        
    def action(self, action):
        """Apply action smoothing."""
        if self.prev_action is not None:
            # Exponential smoothing
            smoothed_action = (self.smoothing_factor * self.prev_action + 
                             (1 - self.smoothing_factor) * action)
        else:
            smoothed_action = action
        
        self.prev_action = action
        return smoothed_action
    
    def reset(self, **kwargs):
        """Reset action history."""
        self.prev_action = None
        return self.env.reset(**kwargs)


class EnvironmentFactory:
    """Factory for creating training environments with various configurations."""
    
    @staticmethod
    def create_env(env_config=None, rank: int = 0, seed: int = 0, 
                   wrappers: list = None, **wrapper_kwargs):
        """Create a single environment instance."""
        
        def _init():
            try:
                # Create base environment
                if CrazyFlieEnv is not None:
                    env = CrazyFlieEnv(config=env_config or EnvConfig())
                else:
                    # Mock environment for testing
                    print("⚠️ Using mock environment - CrazyFlieEnv not available")
                    env = gym.make('CartPole-v1')  # Fallback
                
                # Apply wrappers
                if wrappers:
                    for wrapper_class in wrappers:
                        if wrapper_class == EnhancedRewardWrapper:
                            env = wrapper_class(env, wrapper_kwargs.get('reward_config'))
                        elif wrapper_class == CurriculumWrapper:
                            env = wrapper_class(env, wrapper_kwargs.get('curriculum_config'))
                        elif wrapper_class == ObservationNormalizationWrapper:
                            env = wrapper_class(env, **wrapper_kwargs.get('norm_config', {}))
                        elif wrapper_class == ActionSmoothingWrapper:
                            env = wrapper_class(env, wrapper_kwargs.get('smoothing_factor', 0.8))
                        else:
                            env = wrapper_class(env)
                
                # Monitor wrapper (should be last)
                env = Monitor(env)
                
                # Set seed
                env.reset(seed=seed + rank)
                
                return env
                
            except Exception as e:
                print(f"❌ Error creating environment {rank}: {e}")
                raise
        
        set_random_seed(seed)
        return _init
    
    @staticmethod
    def create_vec_env(num_envs: int, env_config=None, seed: int = 0,
                       wrappers: list = None, use_subprocess: bool = True, **wrapper_kwargs):
        """Create vectorized environment."""
        
        env_fns = [
            EnvironmentFactory.create_env(
                env_config=env_config,
                rank=i,
                seed=seed,
                wrappers=wrappers,
                **wrapper_kwargs
            )
            for i in range(num_envs)
        ]
        
        if use_subprocess and num_envs > 1:
            return SubprocVecEnv(env_fns)
        else:
            return DummyVecEnv(env_fns)
    
    @staticmethod
    def create_training_env(config, enable_live_training: bool = False):
        """Create environment specifically configured for training."""
        
        # Select appropriate wrappers
        wrappers = [EnhancedRewardWrapper]
        wrapper_kwargs = {'reward_config': None}
        
        # Add curriculum learning if specified
        if hasattr(config, 'use_curriculum') and config.use_curriculum:
            wrappers.append(CurriculumWrapper)
        
        # Add normalization for stable training
        wrappers.append(ObservationNormalizationWrapper)
        
        # Add action smoothing for live training (less jerkiness)
        if enable_live_training:
            wrappers.append(ActionSmoothingWrapper)
            wrapper_kwargs['smoothing_factor'] = 0.9
        
        return EnvironmentFactory.create_vec_env(
            num_envs=config.num_envs,
            env_config=config.env_config,
            seed=config.seed,
            wrappers=wrappers,
            use_subprocess=config.num_envs > 1,
            **wrapper_kwargs
        )
    
    @staticmethod
    def create_eval_env(config):
        """Create environment for evaluation."""
        return EnvironmentFactory.create_vec_env(
            num_envs=1,
            env_config=config.env_config,
            seed=config.seed + 1000,  # Different seed for evaluation
            wrappers=[EnhancedRewardWrapper, Monitor],
            use_subprocess=False
        )