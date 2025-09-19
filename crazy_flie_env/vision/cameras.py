# crazy_flie_env/vision/cameras.py
import numpy as np
import mujoco
from typing import Dict, Optional, Tuple

from ..utils.config import EnvConfig
from ..utils.math_utils import (
    quat_to_euler, look_at_quaternion, rotate_vector_by_quat, 
    smooth_interpolate, normalize_angle
)


class CameraSystem:
    """
    Advanced camera system for drone environment.
    
    Manages multiple camera views:
    - Drone FPV camera (first-person view)
    - Chase camera (follows drone smoothly)
    - Overview camera (scene-wide view)
    - Custom positioned cameras
    """
    
    def __init__(self, config: EnvConfig, model: mujoco.MjModel):
        self.config = config
        self.model = model
        
        # Camera IDs from model
        self.camera_ids = {}
        self.renderers = {}
        
        # Chase camera state
        self.chase_state = {
            'distance': config.chase_distance,
            'elevation': config.chase_elevation,
            'height_offset': config.chase_height_offset,
            'initialized': False,
            'lookat': np.zeros(3),
            'smoothed_distance': config.chase_distance
        }
        
        # Initialize cameras
        self._setup_cameras()
        self._create_renderers()
        
        print("✅ Camera system initialized")
        self._print_camera_info()
    
    def _setup_cameras(self):
        """Find and setup cameras in the MuJoCo model."""
        try:
            # Drone FPV camera (attached to drone body)
            self.camera_ids['drone_fpv'] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "drone_fpv"
            )
            print("✅ Found drone FPV camera")
            
        except Exception:
            print("⚠️ Drone FPV camera not found in model")
            self.camera_ids['drone_fpv'] = None
        
        try:
            # Chase/overview camera
            self.camera_ids['overview'] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview"
            )
            print("✅ Found overview camera")
            
        except Exception:
            print("⚠️ Overview camera not found in model")
            self.camera_ids['overview'] = None
        
        # List all available cameras for debugging
        print("📷 Available cameras:")
        for i in range(self.model.ncam):
            cam_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, i
            ) or f"camera_{i}"
            print(f"   Camera {i}: {cam_name}")
    
    def _create_renderers(self):
        """Create MuJoCo renderers for different camera views."""
        # Main drone camera renderer (for observation)
        self.renderers['drone'] = mujoco.Renderer(
            self.model, 
            height=self.config.image_size[1], 
            width=self.config.image_size[0]
        )
        
        # Picture-in-picture renderer (smaller)
        self.renderers['pip'] = mujoco.Renderer(
            self.model,
            height=self.config.pip_size[1],
            width=self.config.pip_size[0]
        )
        
        # Main view renderer (for human rendering)
        self.renderers['main'] = mujoco.Renderer(
            self.model,
            height=self.config.main_view_size[1],
            width=self.config.main_view_size[0]
        )
        
        print(f"✅ Created {len(self.renderers)} renderers")
    
    def get_drone_camera_image(self, data: mujoco.MjData) -> np.ndarray:
        """
        Get image from drone's FPV camera.
        
        Args:
            data: MuJoCo simulation data
            
        Returns:
            RGB image array from drone's perspective
        """
        try:
            if self.camera_ids['drone_fpv'] is not None:
                # Use the drone_fpv camera attached to the drone
                self.renderers['drone'].update_scene(
                    data, camera=self.camera_ids['drone_fpv']
                )
            else:
                # Fallback: create virtual FPV camera
                self._update_virtual_fpv_camera(data)
                self.renderers['drone'].update_scene(data)
            
            # Render and return image
            image = self.renderers['drone'].render()
            return image
            
        except Exception as e:
            print(f"⚠️ Drone camera failed: {e}")
            # Return black image as fallback
            return np.zeros((*self.config.image_size, 3), dtype=np.uint8)
    
    def _update_virtual_fpv_camera(self, data: mujoco.MjData):
        """Create virtual FPV camera when model doesn't have one."""
        # Get drone pose
        drone_pos = data.qpos[0:3].copy()
        drone_quat = data.qpos[3:7].copy()
        
        # Camera offset from drone center (slightly forward)
        camera_offset = np.array([0.1, 0.0, 0.0])  # 10cm forward
        camera_pos = drone_pos + rotate_vector_by_quat(camera_offset, drone_quat)
        
        # Set camera pose (this would require modifying camera in model)
        # For now, this is a placeholder for future implementation
        pass
    
    def update_camera_positions(self, data: mujoco.MjData):
        """Update all dynamic camera positions."""
        self._update_chase_camera(data)
        # Add other camera updates here if needed
    
    def _update_chase_camera(self, data: mujoco.MjData):
        """
        Update chase camera to smoothly follow the drone.
        
        Implements sophisticated camera following with:
        - Smooth distance and elevation changes
        - Look-ahead prediction
        - Collision avoidance (future enhancement)
        """
        if self.camera_ids['overview'] is None:
            return
        
        # Get drone state
        drone_pos = data.qpos[0:3].copy()
        drone_quat = data.qpos[3:7].copy()
        drone_vel = data.qvel[0:3].copy()
        
        # Extract yaw from quaternion
        _, _, yaw = quat_to_euler(drone_quat)
        
        # Chase camera parameters
        distance = self.chase_state['distance']
        elevation_deg = self.chase_state['elevation']
        height_offset = self.chase_state['height_offset']
        
        # Look-at point (slightly above drone, with velocity prediction)
        velocity_prediction = drone_vel * 0.1  # 100ms look-ahead
        lookat_target = drone_pos + velocity_prediction + np.array([0, 0, height_offset])
        
        # Camera position (behind and above drone)
        elevation_rad = np.radians(elevation_deg)
        
        # Calculate camera position in spherical coordinates
        back_direction = np.array([-np.cos(yaw), -np.sin(yaw), 0.0])
        up_component = np.array([0.0, 0.0, -np.tan(elevation_rad)]) * distance
        
        camera_world_pos = lookat_target + distance * back_direction + up_component
        
        # Initialize smoothing state
        if not self.chase_state['initialized']:
            self.chase_state['lookat'] = lookat_target.copy()
            self.chase_state['smoothed_distance'] = distance
            self.chase_state['initialized'] = True
        
        # Smooth camera movement
        alpha = self.config.chase_smoothing_alpha
        self.chase_state['lookat'] = self._smooth_vector(
            self.chase_state['lookat'], lookat_target, alpha
        )
        self.chase_state['smoothed_distance'] = smooth_interpolate(
            self.chase_state['smoothed_distance'], distance, alpha
        )
        
        # Apply smoothed values to camera
        smoothed_distance = self.chase_state['smoothed_distance']
        smoothed_lookat = self.chase_state['lookat']
        
        # Recalculate camera position with smoothed values
        camera_pos = (smoothed_lookat + 
                     smoothed_distance * back_direction + 
                     up_component)
        
        # Update camera in model
        cam_id = self.camera_ids['overview']
        self.model.cam_pos[cam_id] = camera_pos
        self.model.cam_quat[cam_id] = look_at_quaternion(camera_pos, smoothed_lookat)
    
    def _smooth_vector(self, old_vec: np.ndarray, new_vec: np.ndarray, 
                      alpha: float) -> np.ndarray:
        """Smooth vector interpolation."""
        return old_vec * (1 - alpha) + new_vec * alpha
    
    def get_chase_camera_image(self, data: mujoco.MjData) -> np.ndarray:
        """Get image from chase camera view."""
        if self.camera_ids['overview'] is None:
            return self.get_drone_camera_image(data)
        
        try:
            self.renderers['pip'].update_scene(
                data, camera=self.camera_ids['overview']
            )
            return self.renderers['pip'].render()
            
        except Exception as e:
            print(f"⚠️ Chase camera failed: {e}")
            return np.zeros((*self.config.pip_size, 3), dtype=np.uint8)
    
    def get_main_view_image(self, data: mujoco.MjData, camera_name: str = None) -> np.ndarray:
        """Get main view image for human rendering."""
        try:
            if camera_name and camera_name in self.camera_ids:
                camera_id = self.camera_ids[camera_name]
                self.renderers['main'].update_scene(data, camera=camera_id)
            else:
                # Default view
                self.renderers['main'].update_scene(data)
            
            return self.renderers['main'].render()
            
        except Exception as e:
            print(f"⚠️ Main view failed: {e}")
            return np.zeros((*self.config.main_view_size, 3), dtype=np.uint8)
    
    def set_chase_camera_params(self, distance: float = None, elevation: float = None,
                               height_offset: float = None):
        """
        Dynamically adjust chase camera parameters.
        
        Args:
            distance: Distance behind drone (meters)
            elevation: Elevation angle (degrees, negative for looking down)
            height_offset: Height offset above drone (meters)
        """
        if distance is not None:
            self.chase_state['distance'] = max(0.5, distance)  # Minimum distance
        
        if elevation is not None:
            self.chase_state['elevation'] = np.clip(elevation, -89, 89)  # Avoid singularities
        
        if height_offset is not None:
            self.chase_state['height_offset'] = height_offset
        
        print(f"🎥 Chase camera updated: distance={self.chase_state['distance']:.1f}m, "
              f"elevation={self.chase_state['elevation']:.1f}°, "
              f"height_offset={self.chase_state['height_offset']:.1f}m")
    
    def reset_chase_camera(self):
        """Reset chase camera to initial state."""
        self.chase_state['initialized'] = False
        self.chase_state['lookat'] = np.zeros(3)
        self.chase_state['smoothed_distance'] = self.chase_state['distance']
    
    def add_custom_camera(self, name: str, position: np.ndarray, 
                         target: np.ndarray) -> bool:
        """
        Add custom camera position (requires model modification).
        
        Args:
            name: Camera name
            position: Camera position in world coordinates
            target: Target position to look at
            
        Returns:
            True if successfully added, False otherwise
        """
        # This would require dynamically modifying the MuJoCo model
        # For now, this is a placeholder for future implementation
        print(f"⚠️ Custom camera '{name}' not yet supported")
        return False
    
    def get_camera_info(self) -> Dict[str, Dict]:
        """Get information about all cameras."""
        info = {}
        
        for name, cam_id in self.camera_ids.items():
            if cam_id is not None:
                info[name] = {
                    'id': cam_id,
                    'position': self.model.cam_pos[cam_id].copy(),
                    'quaternion': self.model.cam_quat[cam_id].copy(),
                    'fov': self.model.cam_fovy[cam_id] if hasattr(self.model, 'cam_fovy') else 'N/A'
                }
            else:
                info[name] = {'id': None, 'status': 'not_found'}
        
        # Add chase camera state
        info['chase_state'] = self.chase_state.copy()
        
        return info
    
    def _print_camera_info(self):
        """Print camera system information."""
        info = self.get_camera_info()
        print("📷 Camera system status:")
        
        for name, details in info.items():
            if name == 'chase_state':
                continue
                
            if details.get('id') is not None:
                pos = details['position']
                print(f"   {name}: ID={details['id']}, pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            else:
                print(f"   {name}: Not found")
    
    def close(self):
        """Clean up camera resources."""
        for renderer in self.renderers.values():
            try:
                renderer.close()
            except:
                pass
        
        self.renderers.clear()
        print("🔒 Camera system closed")


class CameraController:
    """
    Utility class for controlling camera behavior during episodes.
    
    Provides high-level camera control for different scenarios:
    - Cinematic camera movements
    - Automatic camera switching
    - Recording-friendly camera paths
    """
    
    def __init__(self, camera_system: CameraSystem):
        self.camera_system = camera_system
        self.active_mode = 'chase'
        self.mode_timer = 0.0
        
        # Camera modes configuration
        self.modes = {
            'chase': {'duration': float('inf'), 'switch_to': None},
            'drone_fpv': {'duration': 10.0, 'switch_to': 'chase'},
            'overview': {'duration': 5.0, 'switch_to': 'chase'},
            'cinematic': {'duration': 15.0, 'switch_to': 'chase'}
        }
    
    def update(self, data: mujoco.MjData, dt: float):
        """Update camera controller."""
        self.mode_timer += dt
        
        # Handle automatic mode switching
        current_mode = self.modes[self.active_mode]
        if (current_mode['switch_to'] is not None and 
            self.mode_timer > current_mode['duration']):
            self.switch_mode(current_mode['switch_to'])
        
        # Update camera system
        self.camera_system.update_camera_positions(data)
    
    def switch_mode(self, mode_name: str):
        """Switch camera mode."""
        if mode_name in self.modes:
            self.active_mode = mode_name
            self.mode_timer = 0.0
            print(f"🎥 Camera mode switched to: {mode_name}")
        else:
            print(f"⚠️ Unknown camera mode: {mode_name}")
    
    def get_current_image(self, data: mujoco.MjData) -> np.ndarray:
        """Get image from currently active camera mode."""
        if self.active_mode == 'drone_fpv':
            return self.camera_system.get_drone_camera_image(data)
        elif self.active_mode == 'chase':
            return self.camera_system.get_chase_camera_image(data)
        elif self.active_mode == 'overview':
            return self.camera_system.get_main_view_image(data, 'overview')
        else:
            # Default to drone camera
            return self.camera_system.get_drone_camera_image(data)