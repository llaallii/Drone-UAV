# crazy_flie_env/rewards/reward_functions.py
import numpy as np
import mujoco
from typing import Dict, Any, Callable

from ..utils.config import EnvConfig


class RewardCalculator:
    """
    Flexible reward calculation system for drone navigation tasks.
    
    Supports multiple reward components that can be weighted and combined:
    - Height maintenance rewards
    - Stability bonuses  
    - Action penalties
    - Crash penalties
    - Goal-reaching rewards
    - Efficiency bonuses
    """
    
    def __init__(self, config: EnvConfig):
        self.config = config
        self.reward_components = self._setup_reward_components()
        
        # Tracking for efficiency calculations
        self.prev_position = None
        self.total_distance_traveled = 0.0
        
        print("✅ Reward calculator initialized")
        self._print_reward_summary()
    
    def _setup_reward_components(self) -> Dict[str, Dict[str, Any]]:
        """Setup individual reward components with weights and functions."""
        return {
            'height_maintenance': {
                'weight': self.config.height_reward_weight,
                'function': self._height_maintenance_reward,
                'description': 'Reward for maintaining target altitude'
            },
            'stability': {
                'weight': self.config.stability_reward_weight,
                'function': self._stability_reward,
                'description': 'Reward for low angular velocities'
            },
            'action_penalty': {
                'weight': self.config.action_penalty_weight,
                'function': self._action_penalty,
                'description': 'Penalty for large control actions'
            },
            'crash_penalty': {
                'weight': 1.0,  # Applied as-is when crash occurs
                'function': self._crash_penalty,
                'description': 'Large penalty for crashing'
            },
            'progress': {
                'weight': 0.1,
                'function': self._progress_reward,
                'description': 'Reward for forward progress'
            },
            'efficiency': {
                'weight': 0.05,
                'function': self._efficiency_reward,
                'description': 'Bonus for efficient movement'
            }
        }
    
    def calculate_reward(self, data: mujoco.MjData, action: np.ndarray, 
                        crashed: bool, goal_position: np.ndarray = None) -> float:
        """
        Calculate total reward for current state and action.
        
        Args:
            data: MuJoCo simulation data
            action: Applied action
            crashed: Whether drone has crashed
            goal_position: Optional goal position for navigation tasks
            
        Returns:
            Total reward value
        """
        total_reward = 0.0
        reward_breakdown = {}
        
        # Calculate each reward component
        for component_name, component_info in self.reward_components.items():
            component_reward = component_info['function'](data, action, crashed, goal_position)
            weighted_reward = component_reward * component_info['weight']
            
            total_reward += weighted_reward
            reward_breakdown[component_name] = {
                'raw': component_reward,
                'weighted': weighted_reward
            }
        
        # Store for analysis (optional)
        self._last_reward_breakdown = reward_breakdown
        
        return total_reward
    
    def _height_maintenance_reward(self, data: mujoco.MjData, action: np.ndarray, 
                                  crashed: bool, goal_position: np.ndarray) -> float:
        """Reward for maintaining target height."""
        if crashed:
            return 0.0
        
        current_height = data.qpos[2]
        target_height = self.config.target_height
        
        height_error = abs(current_height - target_height)
        
        # Negative reward increases with distance from target
        return -height_error
    
    def _stability_reward(self, data: mujoco.MjData, action: np.ndarray,
                         crashed: bool, goal_position: np.ndarray) -> float:
        """Reward for stable flight (low angular velocities)."""
        if crashed:
            return 0.0
        
        angular_velocity = data.qvel[3:6]
        angular_speed = np.linalg.norm(angular_velocity)
        
        # Negative reward increases with angular speed
        return float(-angular_speed)
    
    def _action_penalty(self, data: mujoco.MjData, action: np.ndarray,
                       crashed: bool, goal_position: np.ndarray) -> float:
        """Penalty for large actions to encourage efficiency."""
        action_magnitude = np.linalg.norm(action)
        
        # Small penalty for large actions
        return float(-action_magnitude)
    
    def _crash_penalty(self, data: mujoco.MjData, action: np.ndarray,
                      crashed: bool, goal_position: np.ndarray) -> float:
        """Large penalty for crashing."""
        if crashed:
            return self.config.crash_penalty
        return 0.0
    
    def _progress_reward(self, data: mujoco.MjData, action: np.ndarray,
                        crashed: bool, goal_position: np.ndarray) -> float:
        """Reward for making progress toward goal."""
        if crashed or goal_position is None:
            return 0.0
        
        current_position = data.qpos[0:3]
        
        if self.prev_position is not None:
            # Calculate movement toward goal
            prev_dist = np.linalg.norm(self.prev_position - goal_position)
            curr_dist = np.linalg.norm(current_position - goal_position)
            
            progress = prev_dist - curr_dist  # Positive if moving toward goal
            return progress
        
        self.prev_position = current_position.copy()
        return 0.0
    
    def _efficiency_reward(self, data: mujoco.MjData, action: np.ndarray,
                          crashed: bool, goal_position: np.ndarray) -> float:
        """Bonus for efficient movement patterns."""
        if crashed:
            return 0.0
        
        current_position = data.qpos[0:3]
        
        if self.prev_position is not None:
            # Track distance traveled
            distance_moved = np.linalg.norm(current_position - self.prev_position)
            self.total_distance_traveled += distance_moved
            
            # Reward smooth, forward motion
            velocity = data.qvel[0:3]
            speed = np.linalg.norm(velocity)
            
            # Efficiency = speed / (1 + lateral_movement)
            forward_velocity = velocity[0]  # Assuming X is forward
            lateral_velocity = np.sqrt(velocity[1]**2 + velocity[2]**2)
            
            if speed > 0.1:  # Only reward when actually moving
                efficiency = forward_velocity / (1 + lateral_velocity)
                return max(0, efficiency)  # Only positive efficiency
        
        self.prev_position = current_position.copy()
        return 0.0
    
    def reset(self):
        """Reset reward calculator state for new episode."""
        self.prev_position = None
        self.total_distance_traveled = 0.0
        self._last_reward_breakdown = {}
    
    def get_reward_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get detailed breakdown of last reward calculation."""
        return getattr(self, '_last_reward_breakdown', {})
    
    def _print_reward_summary(self):
        """Print summary of reward components."""
        print("🎯 Reward components:")
        for name, info in self.reward_components.items():
            print(f"   {name}: weight={info['weight']:.3f} - {info['description']}")


class NavigationRewardCalculator(RewardCalculator):
    """
    Specialized reward calculator for navigation tasks.
    
    Adds goal-specific rewards and penalties for navigation scenarios.
    """
    
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        
        # Add navigation-specific components
        self.reward_components.update({
            'goal_distance': {
                'weight': -0.01,
                'function': self._goal_distance_penalty,
                'description': 'Penalty for being far from goal'
            },
            'goal_reached': {
                'weight': 10.0,
                'function': self._goal_reached_bonus,
                'description': 'Large bonus for reaching goal'
            },
            'waypoint_progress': {
                'weight': 1.0,
                'function': self._waypoint_progress_reward,
                'description': 'Reward for waypoint progress'
            }
        })
        
        # Navigation state
        self.goal_threshold = 0.5  # meters
        self.waypoints = []
        self.current_waypoint_idx = 0
    
    def set_goal(self, goal_position: np.ndarray):
        """Set navigation goal."""
        self.goal_position = goal_position.copy()
    
    def set_waypoints(self, waypoints: list):
        """Set waypoint sequence for navigation."""
        self.waypoints = [wp.copy() for wp in waypoints]
        self.current_waypoint_idx = 0
    
    def _goal_distance_penalty(self, data: mujoco.MjData, action: np.ndarray,
                              crashed: bool, goal_position: np.ndarray) -> float:
        """Penalty proportional to distance from goal."""
        if crashed or goal_position is None:
            return 0.0
        
        current_position = data.qpos[0:3]
        distance = np.linalg.norm(current_position - goal_position)
        
        return -distance
    
    def _goal_reached_bonus(self, data: mujoco.MjData, action: np.ndarray,
                           crashed: bool, goal_position: np.ndarray) -> float:
        """Large bonus for reaching the goal."""
        if crashed or goal_position is None:
            return 0.0
        
        current_position = data.qpos[0:3]
        distance = np.linalg.norm(current_position - goal_position)
        
        if distance < self.goal_threshold:
            return 1.0  # Large bonus
        
        return 0.0
    
    def _waypoint_progress_reward(self, data: mujoco.MjData, action: np.ndarray,
                                 crashed: bool, goal_position: np.ndarray) -> float:
        """Reward for progressing through waypoints."""
        if crashed or not self.waypoints:
            return 0.0
        
        current_position = data.qpos[0:3]
        
        # Check if current waypoint is reached
        if self.current_waypoint_idx < len(self.waypoints):
            waypoint = self.waypoints[self.current_waypoint_idx]
            distance = np.linalg.norm(current_position - waypoint)
            
            if distance < self.goal_threshold:
                # Reached waypoint - give bonus and advance
                self.current_waypoint_idx += 1
                return 2.0  # Waypoint bonus
        
        return 0.0


class ObstacleAvoidanceRewardCalculator(RewardCalculator):
    """
    Specialized reward calculator for obstacle avoidance tasks.
    
    Adds collision avoidance and safe distance rewards.
    """
    
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        
        # Add obstacle avoidance components
        self.reward_components.update({
            'obstacle_distance': {
                'weight': 0.1,
                'function': self._obstacle_distance_reward,
                'description': 'Reward for maintaining safe distance from obstacles'
            },
            'collision_risk': {
                'weight': -1.0,
                'function': self._collision_risk_penalty,
                'description': 'Penalty for high collision risk'
            },
            'safe_speed': {
                'weight': 0.05,
                'function': self._safe_speed_reward,
                'description': 'Reward for appropriate speed near obstacles'
            }
        })
        
        self.safe_distance_threshold = 1.0  # meters
        self.danger_distance_threshold = 0.5  # meters
    
    def _obstacle_distance_reward(self, data: mujoco.MjData, action: np.ndarray,
                                 crashed: bool, goal_position: np.ndarray) -> float:
        """Reward for maintaining safe distance from obstacles."""
        if crashed:
            return 0.0
        
        # This would need actual obstacle detection
        # For now, return 0 as placeholder
        return 0.0
    
    def _collision_risk_penalty(self, data: mujoco.MjData, action: np.ndarray,
                               crashed: bool, goal_position: np.ndarray) -> float:
        """Penalty for high collision risk based on proximity and velocity."""
        if crashed:
            return 0.0
        
        # Placeholder - would need obstacle sensing
        return 0.0
    
    def _safe_speed_reward(self, data: mujoco.MjData, action: np.ndarray,
                          crashed: bool, goal_position: np.ndarray) -> float:
        """Reward for slowing down near obstacles."""
        if crashed:
            return 0.0
        
        velocity = data.qvel[0:3]
        speed = np.linalg.norm(velocity)
        
        # This would be enhanced with obstacle proximity
        # For now, just reward moderate speeds
        ideal_speed = 2.0  # m/s
        speed_error = abs(speed - ideal_speed)
        
        return float(-speed_error * 0.1)