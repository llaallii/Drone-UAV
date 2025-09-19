# crazy_flie_env/physics/controller.py
import numpy as np
import mujoco
from typing import Dict, Tuple
import math

from ..utils.config import EnvConfig


class PIDController:
    """Generic PID controller implementation."""
    
    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        
        # Anti-windup limits
        self.integral_limit = None
    
    def set_integral_limit(self, limit: float):
        """Set integral anti-windup limits."""
        self.integral_limit = limit
    
    def compute(self, error: float) -> float:
        """Compute PID output."""
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * self.dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term (from error to avoid derivative kick)
        d_term = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        
        return p_term + i_term + d_term
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0


class DroneController:
    """
    Physics-based PID controller for CrazyFlie drone.
    
    Implements cascaded control architecture:
    - Altitude control (thrust)
    - Attitude control (roll, pitch, yaw)
    - Rate control (angular velocities)
    """
    
    def __init__(self, config: EnvConfig, model: mujoco.MjModel):
        self.config = config
        self.model = model
        
        # Calculate physics-based gains
        self.gains = self._calculate_physics_based_gains()
        
        # Initialize PID controllers
        self._init_controllers()
        
        # Get actuator IDs
        self.actuator_ids = self._get_actuator_ids()
        
        print("✅ Drone controller initialized")
        self._print_gain_summary()
    
    def _calculate_physics_based_gains(self) -> Dict[str, Dict[str, float]]:
        """Calculate PID gains based on drone physics and desired performance."""
        
        # === ALTITUDE CONTROL ===
        desired_settling_time = self.config.altitude_settling_time
        desired_overshoot = self.config.altitude_overshoot
        
        # Calculate damping ratio and natural frequency
        overshoot_ratio = desired_overshoot / 100.0
        zeta = -math.log(overshoot_ratio) / math.sqrt(
            math.pi**2 + math.log(overshoot_ratio)**2
        )
        omega_n = 4.0 / (zeta * desired_settling_time)
        
        # Altitude PID gains
        kp_altitude = self.config.mass * omega_n**2
        kd_altitude = self.config.mass * 2 * zeta * omega_n
        ki_altitude = kp_altitude * self.config.altitude_integral_gain
        
        # === ATTITUDE CONTROL ===
        att_settling_time = self.config.attitude_settling_time
        att_overshoot = self.config.attitude_overshoot
        
        att_overshoot_ratio = att_overshoot / 100.0
        att_zeta = -math.log(att_overshoot_ratio) / math.sqrt(
            math.pi**2 + math.log(att_overshoot_ratio)**2
        )
        att_omega_n = 4.0 / (att_zeta * att_settling_time)
        
        # Roll/Pitch gains
        kp_roll_base = self.config.Ixx * att_omega_n**2
        kd_roll_base = self.config.Ixx * 2 * att_zeta * att_omega_n
        
        kp_roll = kp_roll_base * self.config.attitude_scale
        kd_roll = kd_roll_base * self.config.attitude_scale
        ki_roll = kp_roll * self.config.attitude_integral_gain
        
        # Pitch gains (same as roll for symmetric drone)
        kp_pitch = kp_roll
        kd_pitch = kd_roll
        ki_pitch = ki_roll
        
        # Yaw gains (more conservative)
        kp_yaw = self.config.Izz * att_omega_n**2 * self.config.yaw_scale
        kd_yaw = self.config.Izz * 2 * att_zeta * att_omega_n * self.config.yaw_scale
        ki_yaw = kp_yaw * self.config.yaw_integral_gain
        
        return {
            'altitude': {
                'kp': kp_altitude,
                'kd': kd_altitude,
                'ki': ki_altitude
            },
            'attitude': {
                'roll': {'kp': kp_roll, 'kd': kd_roll, 'ki': ki_roll},
                'pitch': {'kp': kp_pitch, 'kd': kd_pitch, 'ki': ki_pitch},
                'yaw': {'kp': kp_yaw, 'kd': kd_yaw, 'ki': ki_yaw}
            }
        }
    
    def _init_controllers(self):
        """Initialize PID controllers."""
        dt = self.config.dt
        
        # Altitude controller
        alt_gains = self.gains['altitude']
        self.altitude_controller = PIDController(
            alt_gains['kp'], alt_gains['ki'], alt_gains['kd'], dt
        )
        self.altitude_controller.set_integral_limit(0.5)
        
        # Attitude controllers
        att_gains = self.gains['attitude']
        
        self.roll_controller = PIDController(
            att_gains['roll']['kp'], att_gains['roll']['ki'], att_gains['roll']['kd'], dt
        )
        self.roll_controller.set_integral_limit(0.1)
        
        self.pitch_controller = PIDController(
            att_gains['pitch']['kp'], att_gains['pitch']['ki'], att_gains['pitch']['kd'], dt
        )
        self.pitch_controller.set_integral_limit(0.1)
        
        self.yaw_controller = PIDController(
            att_gains['yaw']['kp'], att_gains['yaw']['ki'], att_gains['yaw']['kd'], dt
        )
        self.yaw_controller.set_integral_limit(0.05)
    
    def _get_actuator_ids(self) -> Dict[str, int]:
        """Get actuator IDs from model."""
        actuator_ids = {}
        
        try:
            actuator_ids['thrust'] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "body_thrust"
            )
            actuator_ids['x_moment'] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x_moment"
            )
            actuator_ids['y_moment'] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y_moment"
            )
            actuator_ids['z_moment'] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "z_moment"
            )
        except Exception as e:
            print(f"⚠️ Warning: Actuator setup failed: {e}")
            actuator_ids = {}
        
        return actuator_ids
    
    def apply_control(self, data: mujoco.MjData, action: np.ndarray):
        """Apply control action to drone."""
        roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd = action
        
        # Get current state
        current_pos = data.qpos[0:3].copy()
        current_vel = data.qvel[0:3].copy()
        current_quat = data.qpos[3:7].copy()
        current_ang_vel = data.qvel[3:6].copy()
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = self._quat_to_euler(current_quat)
        
        # === ALTITUDE CONTROL ===
        target_height = self.config.initial_height + thrust_cmd * 2.0  # 0.3m to 2.3m range
        current_height = current_pos[2]
        height_error = target_height - current_height
        
        thrust_force = (self.config.hover_thrust + 
                       self.altitude_controller.compute(height_error))
        thrust_force = np.clip(thrust_force, *self.config.thrust_range)
        
        # === ATTITUDE CONTROL ===
        # Scale commands to reasonable limits
        roll_target = roll_cmd * self.config.max_angle
        pitch_target = pitch_cmd * self.config.max_angle
        yaw_rate_target = yaw_rate_cmd * self.config.max_yaw_rate
        
        # Roll control
        roll_error = roll_target - roll
        roll_torque = self.roll_controller.compute(roll_error)
        
        # Pitch control
        pitch_error = pitch_target - pitch
        pitch_torque = self.pitch_controller.compute(pitch_error)
        
        # Yaw rate control (not position control)
        yaw_rate_error = yaw_rate_target - current_ang_vel[2]
        yaw_torque = self.yaw_controller.compute(yaw_rate_error)
        
        # Limit torques for safety
        roll_torque = np.clip(roll_torque, -self.config.max_torque, self.config.max_torque)
        pitch_torque = np.clip(pitch_torque, -self.config.max_torque, self.config.max_torque)
        yaw_torque = np.clip(yaw_torque, -self.config.max_torque, self.config.max_torque)
        
        # Apply to MuJoCo actuators
        self._set_actuator_commands(data, thrust_force, roll_torque, pitch_torque, yaw_torque)
    
    def _set_actuator_commands(self, data: mujoco.MjData, thrust: float, 
                              roll_torque: float, pitch_torque: float, yaw_torque: float):
        """Set actuator commands in MuJoCo."""
        try:
            if self.actuator_ids:
                data.ctrl[self.actuator_ids['thrust']] = thrust
                data.ctrl[self.actuator_ids['x_moment']] = roll_torque
                data.ctrl[self.actuator_ids['y_moment']] = pitch_torque
                data.ctrl[self.actuator_ids['z_moment']] = yaw_torque
            else:
                # Fallback: direct force application
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[2] = thrust
                data.qfrc_applied[3:6] = [roll_torque, pitch_torque, yaw_torque]
                
        except Exception as e:
            print(f"⚠️ Control application failed: {e}")
    
    def _quat_to_euler(self, quat: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion [w,x,y,z] to Euler angles [roll,pitch,yaw]."""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def reset(self):
        """Reset all controller states."""
        self.altitude_controller.reset()
        self.roll_controller.reset()
        self.pitch_controller.reset()
        self.yaw_controller.reset()
    
    def _print_gain_summary(self):
        """Print controller gain summary."""
        print(f"🎛️ Controller gains:")
        print(f"   Altitude: Kp={self.gains['altitude']['kp']:.3f}, "
              f"Ki={self.gains['altitude']['ki']:.3f}, Kd={self.gains['altitude']['kd']:.3f}")
        print(f"   Roll: Kp={self.gains['attitude']['roll']['kp']:.6f}")
        print(f"   Yaw: Kp={self.gains['attitude']['yaw']['kp']:.6f}")
    
    def get_control_diagnostics(self, data: mujoco.MjData) -> Dict[str, float]:
        """Get controller diagnostic information."""
        current_pos = data.qpos[0:3].copy()
        current_quat = data.qpos[3:7].copy()
        roll, pitch, yaw = self._quat_to_euler(current_quat)
        
        return {
            'altitude': current_pos[2],
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'altitude_integral': self.altitude_controller.integral,
            'roll_integral': self.roll_controller.integral,
            'pitch_integral': self.pitch_controller.integral,
            'yaw_integral': self.yaw_controller.integral
        }