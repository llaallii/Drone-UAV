# crazy_flie_env/physics/dynamics.py
import mujoco
import numpy as np
from typing import Optional
from ..utils.config import EnvConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class PhysicsEngine:
    """
    Handles MuJoCo physics simulation and model management.
    
    Responsibilities:
    - Load and validate MuJoCo model
    - Manage simulation stepping
    - Provide access to physics data
    - Handle model resets
    """
    
    def __init__(self, config: EnvConfig):
        self.config = config
        self.model = None
        self.data = None
        self.drone_body_id: Optional[int] = None
        
        self._load_model()
        self._find_drone_body()
        
    def _load_model(self):
        """Load MuJoCo model and initialize data."""
        try:
            # Use the correct MuJoCo API for loading models from XML
            if hasattr(mujoco, "MjModel") and hasattr(mujoco.MjModel, "from_xml_path"):  # type: ignore
                self.model = mujoco.MjModel.from_xml_path(self.config.xml_path)  # type: ignore
                self.data = mujoco.MjData(self.model)  # type: ignore
            elif hasattr(mujoco, "load_model_from_path"):
                self.model = mujoco.load_model_from_path(self.config.xml_path)  # type: ignore
                self.data = mujoco.MjData(self.model)  # type: ignore
            else:
                raise ImportError("Could not find a valid MuJoCo model loading method.")

            logger.info(f"Model loaded: {self.config.xml_path}")
            logger.info(f"Model info: {self.model.nbody} bodies, {self.model.nq} DOF")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info(f"File exists: {self.config.xml_path}")
            raise
    
    def _find_drone_body(self):
        """Find and validate drone body in the model."""
        try:
            # Use the correct MuJoCo API for getting body id by name
            if hasattr(mujoco, 'mj_name2id'):
                self.drone_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cf2')  # type: ignore
            else:
                # Fallback: search manually
                self.drone_body_id = None
                for i in range(self.model.nbody): # type: ignore
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)  # type: ignore
                    if name == 'cf2':
                        self.drone_body_id = i
                        break
                if self.drone_body_id is None:
                    raise ValueError("Could not find drone body 'cf2'.")
            logger.info(f"Found drone body ID: {self.drone_body_id}")
            
        except Exception:
            # List available bodies for debugging
            logger.info("Available bodies:")
            for i in range(self.model.nbody): # type: ignore
                body_name = mujoco.mj_id2name(  # type: ignore
                    self.model, mujoco.mjtObj.mjOBJ_BODY, i  # type: ignore
                ) or f"body_{i}"
                logger.info(f"   Body {i}: {body_name}")
            
            raise ValueError("Could not find drone body 'cf2'. Check available bodies above.")
    
    def reset(self):
        """Reset simulation to initial state."""
        # Reset MuJoCo data
        mujoco.mj_resetData(self.model, self.data)  # type: ignore
        
        # Set initial drone state
        self._set_initial_pose()
        self._set_initial_velocity()
        
        # Forward simulation to update derived quantities
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        
    def _set_initial_pose(self):
        """Set initial drone position and orientation."""
        # Position
        self.data.qpos[0:3] = self.config.spawn_position # type: ignore
        
        # Orientation (quaternion)
        self.data.qpos[3:7] = self.config.spawn_orientation # type: ignore
        
    def _set_initial_velocity(self):
        """Set initial velocities to zero."""
        self.data.qvel[:] = 0.0 # type: ignore
        
    def step(self):
        """Advance physics simulation by one control step."""
        # Run multiple physics steps for smoother control
        for _ in range(self.config.physics_steps):
            mujoco.mj_step(self.model, self.data)  # type: ignore
    
    def get_drone_position(self) -> np.ndarray:
        """Get current drone position."""
        return self.data.qpos[0:3].copy()  # type: ignore
    
    def get_drone_velocity(self) -> np.ndarray:
        """Get current drone velocity."""
        return self.data.qvel[0:3].copy()  # type: ignore

    def get_drone_orientation(self) -> np.ndarray:
        """Get current drone orientation (quaternion)."""
        return self.data.qpos[3:7].copy() # type: ignore
    
    def get_drone_angular_velocity(self) -> np.ndarray:
        """Get current drone angular velocity."""
        return self.data.qvel[3:6].copy()  # type: ignore
    
    def apply_force(self, force: np.ndarray, body_id: Optional[int] = None):
        """Apply external force to drone body."""
        if body_id is None:
            body_id = self.drone_body_id
            
        # Apply force (this would be added to qfrc_applied)
        if len(force) >= 3:
            self.data.qfrc_applied[0:3] = force[0:3] # type: ignore
    
    def apply_torque(self, torque: np.ndarray, body_id: Optional[int] = None):
        """Apply external torque to drone body."""
        if body_id is None:
            body_id = self.drone_body_id
            
        # Apply torque 
        if len(torque) >= 3:
            self.data.qfrc_applied[3:6] = torque[0:3] # type: ignore
    
    def get_actuator_ids(self):
        """Get actuator IDs for drone control."""
        actuator_ids = {}
        
        try:
            actuator_ids['thrust'] = mujoco.mj_name2id(  # type: ignore
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "body_thrust"  # type: ignore
            )
            actuator_ids['x_moment'] = mujoco.mj_name2id(  # type: ignore
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x_moment"  # type: ignore
            )
            actuator_ids['y_moment'] = mujoco.mj_name2id(  # type: ignore
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y_moment"  # type: ignore
            )
            actuator_ids['z_moment'] = mujoco.mj_name2id(  # type: ignore
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "z_moment"  # type: ignore
            )
            
        except Exception as e:
            logger.warning(f"Warning: Could not find some actuators: {e}")
            # List available actuators for debugging
            logger.info("Available actuators:")
            for i in range(self.model.nu): # type: ignore
                name = mujoco.mj_id2name(  # type: ignore
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i  # type: ignore
                ) or f"actuator_{i}"
                logger.info(f"   {i}: {name}")
        
        return actuator_ids
    
    def set_actuator_control(self, actuator_id: int, value: float):
        """Set control value for specific actuator."""
        if 0 <= actuator_id < len(self.data.ctrl): # type: ignore
            self.data.ctrl[actuator_id] = value # type: ignore
        else:
            raise ValueError(f"Invalid actuator ID: {actuator_id}")
    
    def close(self):
        """Clean up resources."""
        # MuJoCo handles its own cleanup
        self.model = None
        self.data = None
        logger.info("Physics engine closed")
    
    @property
    def simulation_time(self) -> float:
        """Current simulation time."""
        return self.data.time if self.data else 0.0
    
    @property
    def is_valid(self) -> bool:
        """Check if physics engine is properly initialized."""
        return self.model is not None and self.data is not None


class PhysicsValidator:
    """Utility class for validating physics simulation state."""
    
    @staticmethod
    def check_stability(data, threshold: float = 1e6) -> bool:  # data: mujoco.MjData
        """Check if simulation is stable (no NaN/Inf values)."""
        # Check positions
        if not np.all(np.isfinite(data.qpos)):
            return False
            
        # Check velocities
        if not np.all(np.isfinite(data.qvel)):
            return False
            
        # Check for excessive values
        if np.any(np.abs(data.qpos) > threshold):
            return False
            
        if np.any(np.abs(data.qvel) > threshold):
            return False
            
        return True
    
    @staticmethod
    def get_physics_diagnostics(data) -> dict:  # data: mujoco.MjData
        """Get diagnostic information about physics state."""
        return {
            'time': data.time,
            'position_norm': np.linalg.norm(data.qpos),
            'velocity_norm': np.linalg.norm(data.qvel),
            'energy': data.energy[0] if hasattr(data, 'energy') else None,
            'stable': PhysicsValidator.check_stability(data)
        }