# crazy_flie_env/utils/config.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple
import os


@dataclass
class EnvConfig:
    """Configuration class for CrazyFlie environment."""
    
    # Model paths
    model_path: str = r"C:\Users\Ratan.Bunkar\Learning\general\rl-agent\Drone-UAV\bitcraze_crazyflie_2"
    @property
    def xml_path(self) -> str:
        return os.path.join(self.model_path, "scene.xml")
    
    # Simulation parameters
    dt: float = 0.02  # 50Hz control loop
    physics_steps: int = 10  # Sub-steps per control step
    max_episode_steps: int = 100000
    
    # Spawn parameters
    initial_height: float = 0.3  # 30cm
    spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.3)
    spawn_orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w,x,y,z
    
    # Safety limits
    crash_height_threshold: float = 0.01  # 1cm
    max_tilt: float = np.pi/2  # 90 degrees
    boundary_radius: float = 5.0  # 5m from origin
    max_altitude: float = 10.0  # 10m ceiling
    
    # Action space limits
    action_bounds_low: np.ndarray = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
    action_bounds_high: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    max_angle: float = 0.25  # ~14 degrees max tilt
    max_yaw_rate: float = 0.8  # rad/s
    max_torque: float = 0.15  # N⋅m
    thrust_range: Tuple[float, float] = (0.05, 0.8)  # N
    
    # Observation space limits
    state_bounds_low: np.ndarray = np.array([
        -10, -10, 0,      # position bounds (x,y,z)
        -5, -5, -5,       # velocity bounds
        -np.pi, -np.pi, -np.pi,  # orientation bounds
        -10, -10, -10     # angular velocity bounds
    ], dtype=np.float32)
    
    state_bounds_high: np.ndarray = np.array([
        10, 10, 10,       # position bounds
        5, 5, 5,          # velocity bounds  
        np.pi, np.pi, np.pi,     # orientation bounds
        10, 10, 10        # angular velocity bounds
    ], dtype=np.float32)
    
    # Camera parameters
    image_size: Tuple[int, int] = (1280, 720)
    image_channels: int = 3
    
    # Camera positioning
    chase_distance: float = 2.2
    chase_elevation: float = -18  # degrees
    chase_height_offset: float = 0.4
    chase_smoothing_alpha: float = 0.25
    
    # Rendering
    main_view_size: Tuple[int, int] = (1280, 720)
    pip_size: Tuple[int, int] = (1280, 720)
    
    # Control gains (will be calculated based on physics)
    # Altitude control
    altitude_settling_time: float = 2.0
    altitude_overshoot: float = 8.0  # percent
    altitude_integral_gain: float = 0.08
    
    # Attitude control  
    attitude_settling_time: float = 0.6
    attitude_overshoot: float = 8.0  # percent
    attitude_scale: float = 120.0  # Conservative scaling
    attitude_integral_gain: float = 0.002
    
    # Yaw control
    yaw_scale: float = 60.0  # More conservative than attitude
    yaw_integral_gain: float = 0.001
    
    # Reward parameters
    height_reward_weight: float = -0.1
    stability_reward_weight: float = -0.01
    action_penalty_weight: float = -0.001
    crash_penalty: float = -10.0
    target_height: float = 1.5  # meters
    
    # Drone physical parameters (CrazyFlie 2.0)
    mass: float = 0.027  # kg
    Ixx: float = 2.3951e-05  # kg⋅m²
    Iyy: float = 2.3951e-05  # kg⋅m²
    Izz: float = 3.2347e-05  # kg⋅m²
    hover_thrust: float = 0.265  # N (mg)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure paths exist
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"Model file not found: {self.xml_path}")
        
        # Validate numeric parameters
        assert self.dt > 0, "Time step must be positive"
        assert self.max_episode_steps > 0, "Episode length must be positive"
        assert self.initial_height > 0, "Initial height must be positive"
        
        # Validate bounds
        assert len(self.action_bounds_low) == 4, "Action bounds must be 4D"
        assert len(self.state_bounds_low) == 12, "State bounds must be 12D"
        
        print(f"✅ Configuration validated successfully")


@dataclass
class TrainingConfig(EnvConfig):
    """Extended configuration for training scenarios."""
    
    # Training-specific parameters
    curriculum_learning: bool = True
    domain_randomization: bool = True
    controller_noise_std: float = 0.1  # 10% noise
    visual_noise_std: float = 0.05
    
    # Environment variations
    spawn_radius: float = 1.0  # Random spawn within radius
    wind_disturbance_std: float = 0.1  # Wind force noise
    
    # Curriculum stages
    curriculum_stages: int = 5
    obstacles_per_stage: int = 10


@dataclass 
class TestingConfig(EnvConfig):
    """Configuration for testing/evaluation."""
    
    # Testing parameters
    deterministic: bool = True
    fixed_seed: int = 42
    
    # Evaluation metrics
    success_height_tolerance: float = 0.1  # meters
    max_evaluation_episodes: int = 100
    
    # Logging
    log_trajectories: bool = True
    save_videos: bool = True
    video_fps: int = 30