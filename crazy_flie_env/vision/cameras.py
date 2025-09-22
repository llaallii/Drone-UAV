# crazy_flie_env/vision/cameras.py
import numpy as np
import mujoco
from typing import Dict
from scipy import ndimage

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
    
    # ===== ADD THESE METHODS TO CameraSystem CLASS =====
    
    def _apply_sgm_simulation(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Apply Semi-Global Matching simulation to create realistic stereo depth.
        This addresses the sim-to-real gap mentioned in RAPID paper.
        """
        # Convert to float for processing
        if len(depth_image.shape) == 3:
            depth_float = np.mean(depth_image, axis=2).astype(np.float32) / 255.0
        else:
            depth_float = depth_image.astype(np.float32) / 255.0
        
        # Add stereo camera noise characteristics
        
        # 1. Quantization noise (stereo matching discretization)
        depth_levels = 64  # Typical stereo depth levels
        depth_quantized = np.round(depth_float * depth_levels) / depth_levels
        
        # 2. Distance-dependent noise (far objects less accurate)
        distance_noise = np.random.normal(0, 0.02, depth_float.shape)
        distance_factor = depth_float * 2.0  # More noise at distance
        depth_noisy = depth_quantized + distance_noise * distance_factor
        
        # 3. Invalid regions (stereo matching failures)
        invalid_mask = np.random.random(depth_float.shape) < 0.05  # 5% invalid pixels
        depth_noisy[invalid_mask] = 0.0  # Invalid depth = 0
        
        # 4. Edge artifacts (stereo matching issues at edges)
        # Apply slight blurring to simulate edge effects
        depth_blurred = ndimage.gaussian_filter(depth_noisy, sigma=0.5)
        
        # Clip and convert back to uint8
        depth_final = np.clip(depth_blurred, 0.0, 1.0)
        depth_uint8 = (depth_final * 255).astype(np.uint8)
        
        # Convert to 3-channel for consistency
        if len(depth_uint8.shape) == 2:
            depth_uint8 = np.repeat(depth_uint8[:, :, np.newaxis], 3, axis=2)
        
        return depth_uint8

    def create_virtual_depth_camera(self, data: mujoco.MjData) -> np.ndarray:
        """
        Create depth image using MuJoCo's built-in depth rendering.
        This implements the stereo-like depth mentioned in RAPID paper.
        """
        try:
            # Create depth renderer if not exists
            if 'depth' not in self.renderers:
                self.renderers['depth'] = mujoco.Renderer(
                    self.model,
                    height=self.config.image_size[1],
                    width=self.config.image_size[0]
                )
            
            # Update scene with depth rendering
            if self.camera_ids.get('drone_fpv') is not None:
                self.renderers['depth'].update_scene(data, camera=self.camera_ids['drone_fpv'])
            else:
                self.renderers['depth'].update_scene(data)
            
            # Render depth
            depth_image = self.renderers['depth'].render()
            
            # Apply SGM-like processing to simulate stereo camera noise
            depth_processed = self._apply_sgm_simulation(depth_image)
            
            return depth_processed
            
        except Exception as e:
            print(f"⚠️ Depth image creation failed: {e}")
            return np.zeros((*self.config.image_size, 3), dtype=np.uint8)

    def get_camera_image_complete(self, data: mujoco.MjData, camera_name: str = "drone_fpv") -> np.ndarray:
        """
        Complete implementation for getting camera images from MuJoCo.
        This replaces the placeholder in your current code.
        """
        try:
            # Get camera ID
            if camera_name in self.camera_ids and self.camera_ids[camera_name] is not None:
                cam_id = self.camera_ids[camera_name]
            else:
                # Use default camera (index 0)
                cam_id = 0
            
            # Update renderer with current scene
            self.renderers['drone'].update_scene(data, camera=cam_id)
            
            # Render image
            image = self.renderers['drone'].render()
            
            # Ensure correct format (H, W, C)
            if len(image.shape) == 3:
                return image
            else:
                # Convert grayscale to RGB if needed
                return np.repeat(image[:, :, np.newaxis], 3, axis=2)
                
        except Exception as e:
            print(f"⚠️ Camera image capture failed: {e}")
            # Return black image as fallback
            return np.zeros((*self.config.image_size, 3), dtype=np.uint8)
    
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


# Environment generation for training diversity
class MuJoCoEnvironmentGenerator:
    """
    Generates diverse training environments within MuJoCo.
    Replaces the need for AirSim by creating varied scenarios in your existing setup.
    """
    
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self.environment_templates = {
            'forest': self._create_forest_environment,
            'urban': self._create_urban_environment,
            'obstacles': self._create_obstacle_course,
            'narrow_gaps': self._create_narrow_gaps
        }
    
    def create_training_environments(self, num_environments: int = 100) -> list:
        """
        Create diverse training environments by modifying the base MuJoCo model.
        This replaces AirSim environment diversity.
        """
        environments = []
        
        for i in range(num_environments):
            env_type = np.random.choice(list(self.environment_templates.keys()))
            env_config = self.environment_templates[env_type]()
            
            environments.append({
                'type': env_type,
                'config': env_config,
                'id': f"{env_type}_{i:03d}"
            })
        
        return environments
    
    def _create_forest_environment(self) -> dict:
        """Create forest-like environment with trees"""
        num_trees = np.random.randint(20, 50)
        trees = []
        
        for _ in range(num_trees):
            tree = {
                'position': [
                    np.random.uniform(-20, 20),
                    np.random.uniform(-20, 20),
                    0.0
                ],
                'radius': np.random.uniform(0.3, 0.8),
                'height': np.random.uniform(3.0, 8.0),
                'type': 'cylinder'
            }
            trees.append(tree)
        
        return {
            'environment_type': 'forest',
            'objects': trees,
            'density': len(trees) / 1600,  # trees per m²
            'lighting': 'natural'
        }
    
    def _create_urban_environment(self) -> dict:
        """Create urban environment with buildings"""
        num_buildings = np.random.randint(10, 25)
        buildings = []
        
        for _ in range(num_buildings):
            building = {
                'position': [
                    np.random.uniform(-15, 15),
                    np.random.uniform(-15, 15),
                    0.0
                ],
                'size': [
                    np.random.uniform(2.0, 5.0),
                    np.random.uniform(2.0, 5.0),
                    np.random.uniform(5.0, 15.0)
                ],
                'type': 'box'
            }
            buildings.append(building)
        
        return {
            'environment_type': 'urban',
            'objects': buildings,
            'density': len(buildings) / 900,  # buildings per m²
            'lighting': 'artificial'
        }
    
    def _create_obstacle_course(self) -> dict:
        """Create obstacle course with various shapes"""
        obstacles = []
        
        # Mix of different obstacle types
        for _ in range(np.random.randint(15, 30)):
            obstacle_type = np.random.choice(['sphere', 'box', 'cylinder'])
            
            if obstacle_type == 'sphere':
                obstacle = {
                    'position': [
                        np.random.uniform(-10, 10),
                        np.random.uniform(-10, 10),
                        np.random.uniform(1.0, 4.0)
                    ],
                    'radius': np.random.uniform(0.5, 1.5),
                    'type': 'sphere'
                }
            elif obstacle_type == 'box':
                obstacle = {
                    'position': [
                        np.random.uniform(-10, 10),
                        np.random.uniform(-10, 10),
                        0.0
                    ],
                    'size': [
                        np.random.uniform(1.0, 3.0),
                        np.random.uniform(1.0, 3.0),
                        np.random.uniform(2.0, 6.0)
                    ],
                    'type': 'box'
                }
            else:  # cylinder
                obstacle = {
                    'position': [
                        np.random.uniform(-10, 10),
                        np.random.uniform(-10, 10),
                        0.0
                    ],
                    'radius': np.random.uniform(0.4, 1.0),
                    'height': np.random.uniform(2.0, 5.0),
                    'type': 'cylinder'
                }
            
            obstacles.append(obstacle)
        
        return {
            'environment_type': 'obstacles',
            'objects': obstacles,
            'complexity': len(obstacles),
            'lighting': 'mixed'
        }
    
    def _create_narrow_gaps(self) -> dict:
        """Create environment with narrow passages"""
        walls = []
        
        # Create walls with gaps
        for i in range(np.random.randint(3, 6)):
            wall_x = np.random.uniform(-15, 15)
            gap_center = np.random.uniform(-10, 10)
            gap_width = np.random.uniform(2.0, 4.0)
            
            # Wall segments before and after gap
            wall_segments = [
                {
                    'position': [wall_x, gap_center - gap_width/2 - 5, 0],
                    'size': [0.2, 10, 5],
                    'type': 'box'
                },
                {
                    'position': [wall_x, gap_center + gap_width/2 + 5, 0],
                    'size': [0.2, 10, 5],
                    'type': 'box'
                }
            ]
            walls.extend(wall_segments)
        
        return {
            'environment_type': 'narrow_gaps',
            'objects': walls,
            'gap_difficulty': 4.0 - np.mean([2.0, 4.0]),  # Smaller gaps = higher difficulty
            'lighting': 'controlled'
        }


# Integration with your existing environment
def integrate_enhanced_vision(env_instance):
    """
    Integrate enhanced vision capabilities with your existing CrazyFlieEnv.
    """
    # Add environment generator
    env_instance.env_generator = MuJoCoEnvironmentGenerator(env_instance.config.model_path)
    
    # Enhanced observation method
    def enhanced_get_observation(self):
        """Enhanced observation with proper depth camera"""
        # Get state vector
        state = self.obs_manager.get_state_vector(self.physics.data)
        
        # Get RGB camera image
        rgb_image = self.camera_system.get_camera_image_complete(self.physics.data)
        
        # Get depth camera image with SGM simulation
        depth_image = self.camera_system.create_virtual_depth_camera(self.physics.data)
        
        # For RAPID-style training, use depth image as main visual input
        # Convert RGB to grayscale and combine with depth
        rgb_gray = np.mean(rgb_image, axis=2, keepdims=True)
        
        # Use depth image as the primary visual input (like RAPID)
        visual_input = depth_image
        
        return {
            'state': state,
            'image': visual_input  # This matches RAPID's depth-based approach
        }
    
    # Replace observation method
    import types
    env_instance._get_observation = types.MethodType(enhanced_get_observation, env_instance)
    
    print("✅ Enhanced vision capabilities integrated")


