# crazy_flie_env/core/action_space.py
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Optional

from ..utils.config import EnvConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ActionManager:
    """
    Manages action space definition and action processing.
    
    Responsibilities:
    - Define action space bounds
    - Process and validate actions
    - Apply action transformations/clipping
    - Provide action utilities
    """
    
    def __init__(self, config: EnvConfig):
        self.config = config
        self.action_space = self._define_action_space()
        
    def _define_action_space(self) -> spaces.Box:
        """Define the action space: [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd]."""
        
        action_space = spaces.Box(
            low=self.config.action_bounds_low,
            high=self.config.action_bounds_high,
            dtype=np.float32
        )
        
        logger.info("Action space defined:")
        logger.info(f"   Roll command: [{self.config.action_bounds_low[0]}, {self.config.action_bounds_high[0]}]")
        logger.info(f"   Pitch command: [{self.config.action_bounds_low[1]}, {self.config.action_bounds_high[1]}]")
        logger.info(f"   Yaw rate command: [{self.config.action_bounds_low[2]}, {self.config.action_bounds_high[2]}]")
        logger.info(f"   Thrust command: [{self.config.action_bounds_low[3]}, {self.config.action_bounds_high[3]}]")
        
        return action_space
    
    def get_action_space(self) -> spaces.Box:
        """Get the action space."""
        return self.action_space
    
    def process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Process and validate raw action from agent.
        
        Args:
            action: Raw action from agent [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd]
            
        Returns:
            Processed and clipped action
        """
        # Ensure action is numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # Validate action dimensions
        if action.shape != (4,):
            raise ValueError(f"Expected action shape (4,), got {action.shape}")
        
        # Clip action to valid bounds
        clipped_action = np.clip(
            action, 
            self.config.action_bounds_low, 
            self.config.action_bounds_high
        ).astype(np.float32)
        
        return clipped_action
    
    def validate_action(self, action: np.ndarray) -> bool:
        """
        Validate that action is within expected bounds.
        
        Args:
            action: Action to validate
            
        Returns:
            True if action is valid, False otherwise
        """
        if not isinstance(action, np.ndarray):
            return False
            
        if action.shape != (4,):
            return False
        
        # Check bounds
        within_bounds = np.all(
            (action >= self.config.action_bounds_low) & 
            (action <= self.config.action_bounds_high)
        )
        
        # Check for NaN/Inf
        is_finite = np.all(np.isfinite(action))
        
        return bool(within_bounds and is_finite)
    
    def sample_action(self) -> np.ndarray:
        """Sample a random valid action."""
        return self.action_space.sample()
    
    def zero_action(self) -> np.ndarray:
        """Get zero action (hover command)."""
        return np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)  # Hover with mid thrust
    
    def parse_action(self, action: np.ndarray) -> Dict[str, float]:
        """
        Parse action into named components.
        
        Args:
            action: Processed action array
            
        Returns:
            Dictionary with named action components
        """
        return {
            'roll_cmd': float(action[0]),
            'pitch_cmd': float(action[1]),
            'yaw_rate_cmd': float(action[2]),
            'thrust_cmd': float(action[3])
        }
    
    def action_to_physical_commands(self, action: np.ndarray) -> Dict[str, float]:
        """
        Convert normalized action to physical command values.
        
        Args:
            action: Normalized action [-1,1] or [0,1] for thrust
            
        Returns:
            Dictionary with physical command values
        """
        roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd = action
        
        # Convert to physical ranges
        physical_commands = {
            'roll_target': roll_cmd * self.config.max_angle,  # radians
            'pitch_target': pitch_cmd * self.config.max_angle,  # radians
            'yaw_rate_target': yaw_rate_cmd * self.config.max_yaw_rate,  # rad/s
            'thrust_target': self.config.initial_height + thrust_cmd * 2.0  # target height in meters
        }
        
        return physical_commands
    
    def get_action_info(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Get comprehensive information about an action.
        
        Args:
            action: Action to analyze
            
        Returns:
            Dictionary with action analysis
        """
        parsed = self.parse_action(action)
        physical = self.action_to_physical_commands(action)
        
        return {
            'raw_action': action.tolist(),
            'parsed': parsed,
            'physical_commands': physical,
            'is_valid': self.validate_action(action),
            'action_magnitude': float(np.linalg.norm(action)),
            'max_component': float(np.max(np.abs(action))),
            'is_aggressive': float(np.max(np.abs(action[:3]))) > 0.7,  # High attitude commands
            'is_hover_like': float(np.max(np.abs(action[:3]))) < 0.1   # Low attitude commands
        }
    
    def smooth_action_transition(self, prev_action: np.ndarray, new_action: np.ndarray, 
                                alpha: float = 0.8) -> np.ndarray:
        """
        Smooth transition between actions to reduce jerky movements.
        
        Args:
            prev_action: Previous action
            new_action: New target action
            alpha: Smoothing factor [0,1] (higher = more responsive)
            
        Returns:
            Smoothed action
        """
        if prev_action is None:
            return new_action
        
        smoothed = prev_action * (1 - alpha) + new_action * alpha
        
        # Ensure result is within bounds
        return np.clip(smoothed, self.config.action_bounds_low, self.config.action_bounds_high)
    
    def add_action_noise(self, action: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        """
        Add Gaussian noise to action for exploration or robustness testing.
        
        Args:
            action: Base action
            noise_std: Standard deviation of noise
            
        Returns:
            Noisy action (clipped to bounds)
        """
        noise = np.random.normal(0, noise_std, action.shape).astype(np.float32)
        noisy_action = action + noise
        
        return np.clip(noisy_action, self.config.action_bounds_low, self.config.action_bounds_high)


class ActionFilter:
    """
    Optional action filtering for smoother control.
    
    Provides low-pass filtering to reduce high-frequency action changes
    that might cause instability in the physical system.
    """
    
    def __init__(self, cutoff_freq: float = 10.0, dt: float = 0.02):
        """
        Initialize action filter.
        
        Args:
            cutoff_freq: Cutoff frequency in Hz
            dt: Time step in seconds
        """
        self.dt = dt
        self.alpha = dt / (dt + 1.0 / (2 * np.pi * cutoff_freq))
        self.prev_action = None
    
    def filter(self, action: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to action."""
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action
        
        # Low-pass filter: y[n] = α*x[n] + (1-α)*y[n-1]
        filtered = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = filtered.copy()
        
        return filtered
    
    def reset(self):
        """Reset filter state."""
        self.prev_action = None


class ActionScheduler:
    """
    Utility for scheduling action changes over time.
    
    Useful for testing or creating scripted behaviors.
    """
    
    def __init__(self):
        self.schedule = []
        self.current_time = 0.0
    
    def add_action(self, time: float, action: np.ndarray, duration: float = 1.0):
        """
        Add timed action to schedule.
        
        Args:
            time: Start time for action
            action: Action to execute
            duration: Duration to hold action
        """
        self.schedule.append({
            'start_time': time,
            'end_time': time + duration,
            'action': action.copy()
        })
        
        # Keep schedule sorted by start time
        self.schedule.sort(key=lambda x: x['start_time'])
    
    def get_action(self, current_time: float, default_action: 'Optional[np.ndarray]' = None) -> np.ndarray:
        """
        Get scheduled action for current time.
        
        Args:
            current_time: Current simulation time
            default_action: Action to return if no scheduled action
            
        Returns:
            Appropriate action for current time
        """
        self.current_time = current_time
        
        # Find active scheduled action
        for scheduled in self.schedule:
            if scheduled['start_time'] <= current_time <= scheduled['end_time']:
                return scheduled['action']
        
        # Return default if no scheduled action
        if default_action is not None:
            return default_action
        else:
            return np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)  # Hover
    
    def clear_schedule(self):
        """Clear all scheduled actions."""
        self.schedule.clear()