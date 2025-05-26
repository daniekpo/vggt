import os
import json
import cv2
import numpy as np


def get_path(*path_segments: str):
    path_str = os.path.join(*path_segments)
    if '.' in os.path.basename(path_str):
        os.makedirs(os.path.dirname(path_str), exist_ok=True)
    else:
        os.makedirs(path_str, exist_ok=True)
    return path_str

def read_file(path, is_json=False, separator=None):
    if is_json:
        with open(path, 'r') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            if separator is None:
                return f.read()
            else:
                return f.read().split(separator)

def create_homogenous_matrix(position, rotation_vec):
    if type(rotation_vec) is not np.ndarray:
        rotation_vec = np.array(rotation_vec)
    if type(position) is not np.ndarray:
        position = np.array(position)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rotation_vec)[0]
    T[:3, 3] = position
    return T

def extract_pos_from_matrix(T):
    position = T[:3, 3]
    rotation_vec = cv2.Rodrigues(T[:3, :3])[0].flatten()
    return np.concatenate([position, rotation_vec])

def get_euler_angles_from_matrix(rotation_matrix):
    """Convert rotation matrix to Euler angles (in degrees)"""
    # Using rotation matrix to get Euler angles (roll, pitch, yaw)
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0
        
    return np.array([roll, pitch, yaw]) * 180 / np.pi  # Convert to degrees