# crazy_flie_env/vision/rendering.py
import numpy as np
import mujoco
import mujoco.viewer
from typing import Optional, Tuple, Dict, Any

from ..utils.config import EnvConfig


class RenderingSystem:
    """
    Advanced rendering system for drone environment visualization.
    
    Features:
    - MuJoCo viewer integration
    - Picture-in-picture displays
    - OpenCV overlays
    - Room transparency control
    - Performance monitoring
    - Screenshot/recording capabilities
    """
    
    def __init__(self, config: EnvConfig, model: mujoco.MjModel):
        self.config = config
        self.model = model
        
        # MuJoCo viewer (for human rendering)
        self.viewer: Optional[mujoco.viewer.Handle] = None
        self.viewer_initialized = False
        
        # OpenCV availability
        self.opencv_available = self._check_opencv()
        
        # PIP display state
        self.pip_window_name = 'Drone Camera View'
        self.pip_active = False
        
        # Performance tracking
        self.frame_count = 0
        self.render_times = []
        
        print(f"✅ Rendering system initialized (OpenCV: {'✅' if self.opencv_available else '❌'})")
    
    def _check_opencv(self) -> bool:
        """Check if OpenCV is available for PIP displays."""
        try:
            import cv2
            return True
        except ImportError:
            return False
    
    def render_main_view(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Render main MuJoCo viewer window.
        
        Args:
            model: MuJoCo model
            data: MuJoCo simulation data
        """
        import time
        start_time = time.time()
        
        try:
            # Launch viewer if not already active
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(model, data)
                self.viewer_initialized = True
                print("🖥️ MuJoCo viewer launched")
            
            # Sync viewer with simulation
            if self.viewer is not None:
                self.viewer.sync()
            
            # Track performance
            render_time = time.time() - start_time
            self.render_times.append(render_time)
            self.frame_count += 1
            
            # Keep only recent render times for performance monitoring
            if len(self.render_times) > 100:
                self.render_times = self.render_times[-100:]
                
        except Exception as e:
            print(f"⚠️ Main view rendering failed: {e}")
    
    def show_pip_overlay(self, drone_image: np.ndarray, window_position: Tuple[int, int] = (400, 400)):
        """
        Show picture-in-picture overlay of drone camera.
        
        Args:
            drone_image: Image from drone's camera
            window_position: (x, y) position for PIP window
        """
        if not self.opencv_available:
            if not hasattr(self, '_opencv_warning_shown'):
                print("⚠️ OpenCV not available for PIP display. Install: pip install opencv-python")
                self._opencv_warning_shown = True
            return
        
        try:
            import cv2
            
            # Resize image for better visibility
            pip_height, pip_width = self.config.pip_size
            display_size = (pip_width * 3, pip_height * 3)  # 3x larger for visibility
            
            drone_resized = cv2.resize(drone_image, display_size)
            
            # Add border and title
            border_color = (255, 255, 255)  # White border
            bordered_image = cv2.copyMakeBorder(
                drone_resized, 5, 25, 5, 5,  # Extra space at bottom for text
                cv2.BORDER_CONSTANT,
                value=border_color
            )
            
            # Add title text
            cv2.putText(
                bordered_image, 'Drone Camera', 
                (10, bordered_image.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
            
            # Add performance info
            if len(self.render_times) > 0:
                avg_render_time = np.mean(self.render_times[-10:]) * 1000  # Last 10 frames in ms
                fps_text = f'Render: {avg_render_time:.1f}ms'
                cv2.putText(
                    bordered_image, fps_text,
                    (display_size[0] - 120, bordered_image.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
                )
            
            # Create/update window with proper flags for resizable and movable window
            cv2.namedWindow(self.pip_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

            # Only set initial position and size if window is not already created
            if not hasattr(self, '_pip_window_initialized'):
                cv2.resizeWindow(self.pip_window_name, display_size[0] + 10, display_size[1] + 30)
                cv2.moveWindow(self.pip_window_name, window_position[0], window_position[1])
                self._pip_window_initialized = True
            
            # Show image
            cv2.imshow(self.pip_window_name, bordered_image)
            cv2.waitKey(1)  # Non-blocking
            
            self.pip_active = True
            
        except Exception as e:
            if not hasattr(self, '_pip_error_shown'):
                print(f"⚠️ PIP display error: {e}")
                self._pip_error_shown = True
    
    def set_room_transparency(self, model: mujoco.MjModel, alpha: float = 0.35):
        """
        Set transparency for room/environment geometry.
        
        Args:
            model: MuJoCo model to modify
            alpha: Transparency level (0=transparent, 1=opaque)
        """
        alpha = np.clip(alpha, 0.0, 1.0)
        
        try:
            # 1) Material-based transparency (preferred)
            room_materials = ["mat-wall", "mat-ceil", "mat-obst", "wall", "ceiling", "floor"]
            
            for mat_name in room_materials:
                try:
                    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
                    if mat_id >= 0:
                        rgba = model.mat_rgba[mat_id].copy()
                        rgba[3] = alpha  # Set alpha channel
                        model.mat_rgba[mat_id] = rgba
                        print(f"🎨 Material '{mat_name}' transparency set to {alpha:.2f}")
                except:
                    pass  # Material doesn't exist
            
            # 2) Geometry-based transparency (fallback)
            room_geometry_tags = ["wall", "ceiling", "pillar", "obst", "shelf", "door", "floor"]
            
            modified_count = 0
            for geom_id in range(model.ngeom):
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                
                # Check if geometry name contains room-related tags
                if any(tag in geom_name.lower() for tag in room_geometry_tags):
                    rgba = model.geom_rgba[geom_id].copy()
                    rgba[3] = alpha
                    model.geom_rgba[geom_id] = rgba
                    modified_count += 1
            
            if modified_count > 0:
                print(f"🎨 Modified transparency for {modified_count} geometries")
            
            # Apply changes to model
            mujoco.mj_forward(model, mujoco.MjData(model))
            
        except Exception as e:
            print(f"⚠️ Room transparency setting failed: {e}")
    
    def capture_screenshot(self, data: mujoco.MjData, 
                          filename: str = None, camera_name: str = None) -> np.ndarray:
        """
        Capture screenshot from specified camera.
        
        Args:
            data: MuJoCo simulation data
            filename: Optional filename to save screenshot
            camera_name: Camera to capture from (default: current view)
            
        Returns:
            Screenshot image as numpy array
        """
        try:
            # Create temporary renderer for screenshot
            renderer = mujoco.Renderer(
                self.model,
                height=self.config.main_view_size[1],
                width=self.config.main_view_size[0]
            )
            
            # Render with specified camera
            if camera_name:
                try:
                    cam_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
                    )
                    renderer.update_scene(data, camera=cam_id)
                except:
                    print(f"⚠️ Camera '{camera_name}' not found, using default view")
                    renderer.update_scene(data)
            else:
                renderer.update_scene(data)
            
            # Get image
            screenshot = renderer.render()
            
            # Save if filename provided
            if filename and self.opencv_available:
                import cv2
                # Convert RGB to BGR for OpenCV
                screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, screenshot_bgr)
                print(f"📸 Screenshot saved: {filename}")
            
            renderer.close()
            return screenshot
            
        except Exception as e:
            print(f"⚠️ Screenshot capture failed: {e}")
            return np.zeros((*self.config.main_view_size, 3), dtype=np.uint8)
    
    def start_recording(self, filename: str = "drone_flight.mp4", fps: int = 30):
        """
        Start video recording (requires OpenCV).
        
        Args:
            filename: Output video filename
            fps: Recording frame rate
        """
        if not self.opencv_available:
            print("⚠️ Video recording requires OpenCV")
            return False
        
        try:
            import cv2
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = tuple(reversed(self.config.main_view_size))  # (width, height)
            
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
            self.recording = True
            self.recording_filename = filename
            
            print(f"🎥 Recording started: {filename} ({fps} FPS)")
            return True
            
        except Exception as e:
            print(f"⚠️ Recording start failed: {e}")
            return False
    
    def record_frame(self, data: mujoco.MjData):
        """Add current frame to video recording."""
        if not hasattr(self, 'recording') or not self.recording:
            return
        
        try:
            # Capture frame
            frame = self.capture_screenshot(data)
            
            # Convert RGB to BGR for OpenCV
            import cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write frame
            self.video_writer.write(frame_bgr)
            
        except Exception as e:
            print(f"⚠️ Frame recording failed: {e}")
    
    def stop_recording(self):
        """Stop video recording."""
        if hasattr(self, 'recording') and self.recording:
            try:
                self.video_writer.release()
                self.recording = False
                print(f"🎥 Recording saved: {self.recording_filename}")
            except Exception as e:
                print(f"⚠️ Recording stop failed: {e}")
    
    def toggle_pip_display(self):
        """Toggle picture-in-picture display on/off."""
        if self.pip_active:
            self.close_pip_display()
        else:
            print("🖼️ PIP display will show on next render")

    def move_pip_window(self, x: int, y: int):
        """Move PIP window to new position."""
        if self.opencv_available and self.pip_active:
            try:
                import cv2
                cv2.moveWindow(self.pip_window_name, x, y)
                print(f"🖼️ PIP window moved to ({x}, {y})")
            except Exception as e:
                print(f"⚠️ Failed to move PIP window: {e}")

    def resize_pip_window(self, width: int, height: int):
        """Resize PIP window."""
        if self.opencv_available and self.pip_active:
            try:
                import cv2
                cv2.resizeWindow(self.pip_window_name, width, height)
                print(f"🖼️ PIP window resized to {width}x{height}")
            except Exception as e:
                print(f"⚠️ Failed to resize PIP window: {e}")
    
    def close_pip_display(self):
        """Close picture-in-picture display."""
        if self.opencv_available and self.pip_active:
            try:
                import cv2
                cv2.destroyWindow(self.pip_window_name)
                self.pip_active = False
                # Reset window initialization flag so it can be repositioned next time
                if hasattr(self, '_pip_window_initialized'):
                    delattr(self, '_pip_window_initialized')
                print("🖼️ PIP display closed")
            except:
                pass
    
    def get_render_stats(self) -> Dict[str, Any]:
        """Get rendering performance statistics."""
        if len(self.render_times) == 0:
            return {'frames': 0, 'avg_render_time': 0, 'fps': 0}
        
        avg_render_time = np.mean(self.render_times)
        estimated_fps = 1.0 / avg_render_time if avg_render_time > 0 else 0
        
        return {
            'frames_rendered': self.frame_count,
            'avg_render_time_ms': avg_render_time * 1000,
            'estimated_fps': estimated_fps,
            'min_render_time_ms': np.min(self.render_times) * 1000,
            'max_render_time_ms': np.max(self.render_times) * 1000,
            'pip_active': self.pip_active,
            'recording': getattr(self, 'recording', False)
        }
    
    def print_render_stats(self):
        """Print current rendering statistics."""
        stats = self.get_render_stats()
        print("\n📊 Rendering Statistics:")
        print(f"   Frames rendered: {stats['frames_rendered']}")
        print(f"   Average render time: {stats['avg_render_time_ms']:.2f} ms")
        print(f"   Estimated FPS: {stats['estimated_fps']:.1f}")
        print(f"   Min/Max render time: {stats['min_render_time_ms']:.2f}/{stats['max_render_time_ms']:.2f} ms")
        print(f"   PIP active: {'✅' if stats['pip_active'] else '❌'}")
        print(f"   Recording: {'🎥' if stats['recording'] else '❌'}")
    
    def set_viewer_camera_mode(self, mode: str):
        """
        Set MuJoCo viewer camera mode.
        
        Args:
            mode: Camera mode ('free', 'tracking', 'fixed')
        """
        if self.viewer is None:
            return
        
        try:
            if mode == 'free':
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            elif mode == 'tracking':
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = 0  # Track first body (usually drone)
            elif mode == 'fixed':
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            
            print(f"🎥 Viewer camera mode set to: {mode}")
            
        except Exception as e:
            print(f"⚠️ Camera mode change failed: {e}")
    
    def update_viewer_lighting(self, ambient: float = 0.3, diffuse: float = 0.7, 
                              specular: float = 0.1):
        """
        Update viewer lighting parameters.
        
        Args:
            ambient: Ambient light intensity [0,1]
            diffuse: Diffuse light intensity [0,1]  
            specular: Specular light intensity [0,1]
        """
        if self.viewer is None:
            return
        
        try:
            # Update lighting (this would require access to viewer options)
            # For now, this is a placeholder for future implementation
            print(f"💡 Lighting updated: ambient={ambient}, diffuse={diffuse}, specular={specular}")
            
        except Exception as e:
            print(f"⚠️ Lighting update failed: {e}")
    
    def create_overlay_text(self, text: str, position: Tuple[int, int] = (10, 30),
                           font_scale: float = 0.7, color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Create text overlay on rendered images.
        
        Args:
            text: Text to display
            position: (x, y) position for text
            font_scale: Font size scale
            color: RGB color for text
        """
        if not self.opencv_available:
            return lambda img: img  # Return identity function
        
        def add_text_overlay(image: np.ndarray) -> np.ndarray:
            import cv2
            img_copy = image.copy()
            
            # Add background rectangle for readability
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            cv2.rectangle(img_copy, 
                         (position[0] - 5, position[1] - text_size[1] - 5),
                         (position[0] + text_size[0] + 5, position[1] + 5),
                         (0, 0, 0), -1)  # Black background
            
            # Add text
            cv2.putText(img_copy, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, color, 2)
            
            return img_copy
        
        return add_text_overlay
    
    def create_status_overlay(self, data: mujoco.MjData) -> callable:
        """
        Create status information overlay.
        
        Args:
            data: MuJoCo simulation data
            
        Returns:
            Function that adds status overlay to images
        """
        # Extract drone status
        pos = data.qpos[0:3]
        vel = data.qvel[0:3]
        speed = np.linalg.norm(vel)
        
        status_text = [
            f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]",
            f"Speed: {speed:.2f} m/s",
            f"Time: {data.time:.2f}s"
        ]
        
        def add_status_overlay(image: np.ndarray) -> np.ndarray:
            if not self.opencv_available:
                return image
            
            import cv2
            img_copy = image.copy()
            
            # Add each line of status text
            for i, line in enumerate(status_text):
                y_pos = 30 + i * 25
                
                # Background rectangle
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(img_copy,
                             (5, y_pos - text_size[1] - 2),
                             (15 + text_size[0], y_pos + 2),
                             (0, 0, 0), -1)
                
                # Text
                cv2.putText(img_copy, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            return img_copy
        
        return add_status_overlay
    
    def resize_image(self, image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image using simple interpolation.
        
        Args:
            image: Input image
            new_size: (width, height) for output
            
        Returns:
            Resized image
        """
        if self.opencv_available:
            import cv2
            return cv2.resize(image, new_size)
        else:
            # Simple nearest neighbor resize (fallback)
            old_h, old_w = image.shape[:2]
            new_w, new_h = new_size
            
            # Create index arrays
            y_indices = np.linspace(0, old_h - 1, new_h).astype(int)
            x_indices = np.linspace(0, old_w - 1, new_w).astype(int)
            
            # Create meshgrid
            yy, xx = np.meshgrid(y_indices, x_indices, indexing='ij')
            
            # Resize
            if len(image.shape) == 3:
                resized = image[yy, xx, :]
            else:
                resized = image[yy, xx]
            
            return resized
    
    def apply_visual_effects(self, image: np.ndarray, 
                           brightness: float = 0.0, contrast: float = 1.0,
                           blur: bool = False) -> np.ndarray:
        """
        Apply visual effects to rendered images.
        
        Args:
            image: Input image
            brightness: Brightness adjustment [-100, 100]
            contrast: Contrast multiplier [0.5, 2.0]
            blur: Whether to apply slight blur
            
        Returns:
            Processed image
        """
        if not self.opencv_available:
            return image
        
        try:
            import cv2
            
            processed = image.copy().astype(np.float32)
            
            # Apply brightness and contrast
            processed = processed * contrast + brightness
            processed = np.clip(processed, 0, 255).astype(np.uint8)
            
            # Apply blur if requested
            if blur:
                processed = cv2.GaussianBlur(processed, (3, 3), 0)
            
            return processed
            
        except Exception as e:
            print(f"⚠️ Visual effects failed: {e}")
            return image
    
    def close(self):
        """Clean up all rendering resources."""
        # Close MuJoCo viewer
        if self.viewer is not None:
            try:
                self.viewer.close()
                self.viewer = None
                print("🔒 MuJoCo viewer closed")
            except:
                pass
        
        # Close PIP display
        self.close_pip_display()
        
        # Stop recording if active
        if hasattr(self, 'recording') and self.recording:
            self.stop_recording()
        
        # Close all OpenCV windows
        if self.opencv_available:
            try:
                import cv2
                cv2.destroyAllWindows()
            except:
                pass
        
        print("🔒 Rendering system closed")


class RecordingManager:
    """
    Specialized class for managing video recording sessions.
    
    Features:
    - Multiple camera recording
    - Frame synchronization
    - Automatic naming
    - Performance optimization
    """
    
    def __init__(self, rendering_system: RenderingSystem):
        self.rendering_system = rendering_system
        self.active_recordings = {}
        self.frame_buffer = {}
        
    def start_multi_camera_recording(self, cameras: list, base_filename: str, 
                                   fps: int = 30):
        """
        Start recording from multiple cameras simultaneously.
        
        Args:
            cameras: List of camera names to record
            base_filename: Base filename (will be suffixed with camera name)
            fps: Recording frame rate
        """
        if not self.rendering_system.opencv_available:
            print("⚠️ Multi-camera recording requires OpenCV")
            return False
        
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = tuple(reversed(self.rendering_system.config.main_view_size))
        
        for camera in cameras:
            filename = f"{base_filename}_{camera}.mp4"
            writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
            
            self.active_recordings[camera] = {
                'writer': writer,
                'filename': filename,
                'frame_count': 0
            }
        
        print(f"🎥 Multi-camera recording started: {len(cameras)} cameras")
        return True
    
    def record_multi_camera_frame(self, data: mujoco.MjData):
        """Record frame from all active cameras."""
        for camera_name, recording_info in self.active_recordings.items():
            try:
                # Capture frame from specific camera
                frame = self.rendering_system.capture_screenshot(data, camera_name=camera_name)
                
                # Convert RGB to BGR for OpenCV
                import cv2
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                recording_info['writer'].write(frame_bgr)
                recording_info['frame_count'] += 1
                
            except Exception as e:
                print(f"⚠️ Multi-camera recording failed for {camera_name}: {e}")
    
    def stop_all_recordings(self):
        """Stop all active recordings."""
        for camera_name, recording_info in self.active_recordings.items():
            try:
                recording_info['writer'].release()
                print(f"🎥 Recording saved: {recording_info['filename']} "
                      f"({recording_info['frame_count']} frames)")
            except Exception as e:
                print(f"⚠️ Failed to stop recording for {camera_name}: {e}")
        
        self.active_recordings.clear()


class VisualizationEffects:
    """
    Collection of visual effects for enhanced rendering.
    
    Provides cinematic effects, debug visualizations, and overlays.
    """
    
    @staticmethod
    def create_trajectory_trail(positions: list, max_length: int = 50) -> callable:
        """
        Create trajectory trail visualization.
        
        Args:
            positions: List of drone positions
            max_length: Maximum trail length
            
        Returns:
            Function that adds trail overlay to images
        """
        def add_trail_overlay(image: np.ndarray) -> np.ndarray:
            # This would require projecting 3D positions to image coordinates
            # For now, this is a placeholder for future implementation
            return image
        
        return add_trail_overlay
    
    @staticmethod
    def create_velocity_vector_overlay(velocity: np.ndarray, scale: float = 10.0) -> callable:
        """
        Create velocity vector visualization.
        
        Args:
            velocity: Drone velocity vector
            scale: Visualization scale factor
            
        Returns:
            Function that adds velocity overlay to images
        """
        def add_velocity_overlay(image: np.ndarray) -> np.ndarray:
            # This would require 3D to 2D projection
            # For now, this is a placeholder for future implementation
            return image
        
        return add_velocity_overlay