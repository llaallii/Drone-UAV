# crazy_flie_env/utils/math_utils.py
import numpy as np
from typing import Tuple


def quat_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion [w,x,y,z] to Euler angles [roll,pitch,yaw].
    
    Args:
        quat: Quaternion in [w,x,y,z] format
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
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


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles to quaternion [w,x,y,z].
    
    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
        
    Returns:
        Quaternion in [w,x,y,z] format
    """
    # Half angles
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    # Quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def rotate_vector_by_quat(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by a quaternion.
    
    Args:
        vector: 3D vector to rotate
        quaternion: Quaternion in [w,x,y,z] format
        
    Returns:
        Rotated vector
    """
    w, x, y, z = quaternion
    
    # Quaternion rotation formula
    # v' = v + 2 * r × (r × v + w * v)
    r = np.array([x, y, z])
    v = vector.copy()
    
    cross1 = np.cross(r, v) + w * v
    cross2 = np.cross(r, cross1)
    
    rotated = v + 2 * cross2
    return rotated


from typing import Optional

def look_at_quaternion(eye_pos: np.ndarray, target_pos: np.ndarray, 
                      up: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate quaternion to look from eye_pos to target_pos.
    
    Args:
        eye_pos: Eye position
        target_pos: Target position to look at
        up: Up direction (default: [0,0,1])
        
    Returns:
        Look-at quaternion in [w,x,y,z] format
    """
    if up is None:
        up = np.array([0, 0, 1])
    
    # Calculate forward direction
    forward = target_pos - eye_pos
    forward = forward / np.linalg.norm(forward)
    
    # Calculate right direction
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Recalculate up to ensure orthogonality
    up = np.cross(right, forward)
    
    # Create rotation matrix
    rotation_matrix = np.column_stack([right, up, -forward])
    
    # Convert rotation matrix to quaternion
    return rotation_matrix_to_quat(rotation_matrix)


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion in [w,x,y,z] format
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-π, π] range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in [-π, π]
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def smooth_interpolate(old_value: float, new_value: float, alpha: float) -> float:
    """
    Exponential smoothing interpolation.
    
    Args:
        old_value: Previous value
        new_value: New target value
        alpha: Smoothing factor [0, 1] (higher = more responsive)
        
    Returns:
        Smoothed value
    """
    return old_value * (1 - alpha) + new_value * alpha


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value to [min_val, max_val] range.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def wrap_to_pi(angle: float) -> float:
    """
    Wrap angle to [-π, π] range.
    
    Args:
        angle: Input angle in radians
        
    Returns:
        Wrapped angle in [-π, π]
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.
    
    Args:
        q1: First quaternion [w,x,y,z]
        q2: Second quaternion [w,x,y,z]
        
    Returns:
        Product quaternion [w,x,y,z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Calculate quaternion conjugate.
    
    Args:
        q: Quaternion [w,x,y,z]
        
    Returns:
        Conjugate quaternion [w,-x,-y,-z]
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quaternion_norm(q: np.ndarray) -> float:
    """
    Calculate quaternion norm.
    
    Args:
        q: Quaternion [w,x,y,z]
        
    Returns:
        Quaternion norm (scalar)
    """
    norm = np.linalg.norm(q)
    return float(norm)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion to unit length.
    
    Args:
        q: Quaternion [w,x,y,z]
        
    Returns:
        Normalized unit quaternion
    """
    norm = quaternion_norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    return q / norm


def angular_distance(angle1: float, angle2: float) -> float:
    """
    Calculate shortest angular distance between two angles.
    
    Args:
        angle1: First angle in radians
        angle2: Second angle in radians
        
    Returns:
        Shortest angular distance in radians
    """
    diff = angle2 - angle1
    return wrap_to_pi(diff)


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate angle between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Angle between vectors in radians
    """
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate angle
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(cos_angle)