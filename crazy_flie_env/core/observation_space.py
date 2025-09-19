# crazy_flie_env/core/observation_space.py
import numpy as np
import mujoco
from gymnasium import spaces
from typing import Dict

from ..utils.config import EnvConfig
from ..utils.math_utils import quat_to_euler


class ObservationManager:
    """
    Manages observation space definition and state vector extraction.
    
    Responsibilities:
    - Define observation space (state + image)
    - Extract and process state vectors
    - Handle coordinate transformations
    - Provide clean observation interface
    """
    
    def __init__(self, config: EnvConfig):
        self.config = config
        self.observation_space = self._define_observation_space()
        
    def _define_observation_space(self) -> spaces.Dict:
        """Define the observation space: state vector + camera image."""
        
        # State vector: [x,y,z, vx,vy,vz, roll,pitch,yaw, wx,wy,wz]
        state_space = spaces.Box(
            low=self.config.state_bounds_low,
            high=self.config.state_bounds_high,
            dtype=np.float32
        )
        
        # Camera image: HxWxC
        image_shape = (*self.config.image_size, self.config.image_channels)
        image_space = spaces.Box(
            low=0, 
            high=255, 
            shape=image_shape, 
            dtype=np.uint8
        )
        
        # Combined observation space
        obs_space = spaces.Dict({
            'state': state_space,
            'image': image_space
        })
        
        print(f"✅ Observation space defined:")
        print(f"   State vector: {len(self.config.state_bounds_low)} dimensions")
        print(f"   Camera image: {image_shape}")
        
        return obs_space
    
    def get_observation_space(self) -> spaces.Dict:
        """Get the observation space."""
        return self.observation_space
    
    def get_state_vector(self, data: mujoco.MjData) -> np.ndarray:
        """Extract 12D state vector: [x,y,z, vx,vy,vz, roll,pitch,yaw, wx,wy,wz]."""
        
        # Position (first 3 elements of qpos)
        position = data.qpos[0:3].copy()
        
        # Linear velocity (first 3 elements of qvel)  
        velocity = data.qvel[0:3].copy()
        
        # Orientation: Convert quaternion to Euler angles
        quaternion = data.qpos[3:7].copy()  # [w,x,y,z] format in MuJoCo
        roll, pitch, yaw = quat_to_euler(quaternion)
        
        # Angular velocity (elements 3:6 of qvel)
        angular_velocity = data.qvel[3:6].copy()
        
        # Combine into state vector
        state_vector = np.concatenate([
            position,           # [x, y, z]
            velocity,           # [vx, vy, vz]  
            [roll, pitch, yaw], # [roll, pitch, yaw]
            angular_velocity    # [wx, wy, wz]
        ])
        
        return state_vector.astype(np.float32)
    
    def get_drone_pose(self, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        """Get drone pose information in convenient format."""
        state = self.get_state_vector(data)
        
        return {
            'position': state[0:3],
            'velocity': state[3:6], 
            'orientation': state[6:9],  # Euler angles
            'angular_velocity': state[9:12],
            'quaternion': data.qpos[3:7].copy()  # Original quaternion
        }
    
    def validate_state(self, state: np.ndarray) -> bool:
        """Validate that state vector is within expected bounds."""
        if len(state) != 12:
            return False
            
        # Check bounds
        within_bounds = np.all(
            (state >= self.config.state_bounds_low) & 
            (state <= self.config.state_bounds_high)
        )
        
        # Check for NaN/Inf
        is_finite = np.all(np.isfinite(state))
        
        return within_bounds and is_finite
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state vector to [-1, 1] range."""
        # Normalize each component to [-1, 1]
        normalized = 2 * (state - self.config.state_bounds_low) / (
            self.config.state_bounds_high - self.config.state_bounds_low
        ) - 1
        
        return normalized.astype(np.float32)
    
    def denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """Convert normalized state back to original scale."""
        # Convert from [-1, 1] back to original bounds
        denormalized = (normalized_state + 1) / 2 * (
            self.config.state_bounds_high - self.config.state_bounds_low
        ) + self.config.state_bounds_low
        
        return denormalized.astype(np.float32)
    
    def get_relative_goal_vector(self, data: mujoco.MjData, goal_position: np.ndarray) -> np.ndarray:
        """Calculate relative goal vector from current position."""
        current_position = data.qpos[0:3]
        relative_goal = goal_position - current_position
        return relative_goal.astype(np.float32)
    
    def extract_flight_metrics(self, data: mujoco.MjData) -> Dict[str, float]:
        """Extract useful flight metrics for analysis."""
        state = self.get_state_vector(data)
        
        return {
            'altitude': float(state[2]),
            'speed': float(np.linalg.norm(state[3:6])),
            'angular_speed': float(np.linalg.norm(state[9:12])),
            'tilt_angle': float(np.sqrt(state[6]**2 + state[7]**2)),  # Combined roll/pitch
            'yaw_angle': float(state[8]),
            'distance_from_origin': float(np.linalg.norm(state[0:2]))
        }