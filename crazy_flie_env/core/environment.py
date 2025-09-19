# crazy_flie_env/core/environment.py
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple

from .observation_space import ObservationManager
from .action_space import ActionManager
from ..physics.dynamics import PhysicsEngine
from ..physics.controller import DroneController
from ..vision.cameras import CameraSystem
from ..vision.rendering import RenderingSystem
from ..rewards.reward_functions import RewardCalculator
from ..utils.config import EnvConfig


class CrazyFlieEnv(gym.Env):
    """
    Modular CrazyFlie drone environment for reinforcement learning.
    
    Features:
    - Multi-modal observations (state + vision)
    - Physics-based control with PID controllers
    - Advanced camera system with chase and FPV views
    - Configurable reward functions
    - Real-time visualization
    """
    
    def __init__(self, config: EnvConfig = None):
        super().__init__()
        
        # Use default config if none provided
        self.config = config or EnvConfig()
        
        # Initialize core components
        self._init_components()
        
        # Set up gym spaces
        self.observation_space = self.obs_manager.get_observation_space()
        self.action_space = self.action_manager.get_action_space()
        
        # Episode tracking
        self.step_count = 0
        self._episode_active = False
        
        print(f"✅ CrazyFlie Environment initialized")
        print(f"📊 Observation space: {self.observation_space}")
        print(f"🎮 Action space: {self.action_space}")

    def _init_components(self):
        """Initialize all environment components."""
        # Physics and dynamics
        self.physics = PhysicsEngine(self.config)
        self.controller = DroneController(self.config, self.physics.model)
        
        # Observation and action handling
        self.obs_manager = ObservationManager(self.config)
        self.action_manager = ActionManager(self.config)
        
        # Vision system
        self.camera_system = CameraSystem(self.config, self.physics.model)
        self.renderer = RenderingSystem(self.config, self.physics.model)
        
        # Reward calculation
        self.reward_calc = RewardCalculator(self.config)

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset physics simulation
        self.physics.reset()
        
        # Reset controller state
        self.controller.reset()
        
        # Reset episode tracking
        self.step_count = 0
        self._episode_active = True
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        if not self._episode_active:
            raise RuntimeError("Environment not active. Call reset() first.")
        
        # Process and apply action
        processed_action = self.action_manager.process_action(action)
        self.controller.apply_control(self.physics.data, processed_action)
        
        # Step physics simulation
        self.physics.step()
        
        # Update step counter
        self.step_count += 1
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self.reward_calc.calculate_reward(
            self.physics.data, processed_action, self._check_crash()
        )
        
        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self.step_count >= self.config.max_episode_steps
        
        if terminated or truncated:
            self._episode_active = False
        
        # Get info
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Get state vector
        state = self.obs_manager.get_state_vector(self.physics.data)
        
        # Get camera image
        image = self.camera_system.get_drone_camera_image(self.physics.data)
        
        return {
            'state': state,
            'image': image
        }

    def _get_info(self) -> Dict[str, Any]:
        """Return additional info dictionary."""
        return {
            'step_count': self.step_count,
            'drone_height': self.physics.data.qpos[2],
            'is_crashed': self._check_crash(),
            'episode_active': self._episode_active
        }

    def _check_crash(self) -> bool:
        """Check if drone has crashed."""
        height = self.physics.data.qpos[2]
        
        # Ground collision
        if height < self.config.crash_height_threshold:
            return True
            
        # Severe tilt
        state = self.obs_manager.get_state_vector(self.physics.data)
        roll, pitch = state[6], state[7]
        
        if abs(roll) > self.config.max_tilt or abs(pitch) > self.config.max_tilt:
            return True
            
        return False

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Crashed
        if self._check_crash():
            return True
        
        # Out of bounds
        pos = self.physics.data.qpos[0:3]
        distance_from_origin = np.linalg.norm(pos[:2])
        
        if distance_from_origin > self.config.boundary_radius:
            return True
        
        if pos[2] > self.config.max_altitude:
            return True
            
        return False

    def render(self, mode: str = 'human'):
        """Render the environment."""
        if mode != 'human':
            return self.camera_system.get_drone_camera_image(self.physics.data)
        
        # Launch/update MuJoCo viewer
        self.renderer.render_main_view(self.physics.model, self.physics.data)
        
        # Update camera positions
        self.camera_system.update_camera_positions(self.physics.data)
        
        # Show drone camera PIP
        self.renderer.show_pip_overlay(
            self.camera_system.get_drone_camera_image(self.physics.data)
        )

    def close(self):
        """Clean up resources."""
        self.renderer.close()
        self.camera_system.close()
        self.physics.close()
        print("🔒 Environment closed successfully")

    def set_room_transparency(self, alpha: float = 0.35):
        """Set room transparency for better visualization."""
        self.renderer.set_room_transparency(self.physics.model, alpha)

    # Convenience properties
    @property
    def drone_position(self) -> np.ndarray:
        """Current drone position."""
        return self.physics.data.qpos[0:3].copy()
    
    @property
    def drone_velocity(self) -> np.ndarray:
        """Current drone velocity."""
        return self.physics.data.qvel[0:3].copy()
    
    @property 
    def is_crashed(self) -> bool:
        """Whether drone has crashed."""
        return self._check_crash()