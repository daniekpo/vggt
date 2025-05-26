from typing import Tuple, List
import cv2
import numpy as np
import sys
import os

from utils.robot import Robot

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.camera import Intrinsics, Extrinsics

def pixel_to_camera(pixel, depth: np.ndarray, intrinsics: Intrinsics):
    """Convert pixel coordinates to camera coordinates.
    Assumes depth was scaled by 1000 already.

    pixel: tuple of (x, y)
    depth: numpy array of shape (height, width)
    intrinsics: Intrinsics object
    """
    # TODO: do bilinear interpolation if x and y are not integers
    x_pixel, y_pixel = int(pixel[0]), int(pixel[1])
    z = depth[y_pixel, x_pixel]

    x = (x_pixel - intrinsics.cx) * z / intrinsics.fx
    y = (y_pixel - intrinsics.cy) * z / intrinsics.fy

    return np.array([x, y, z])

def campoint_to_worldpoint(cam_point, extrinsics: Extrinsics, robot_pose):
    cam2ee = extrinsics.data

    if not isinstance(robot_pose, np.ndarray):
        robot_pose = np.array(robot_pose)

    point_h = np.ones(4)
    point_h[:3] = cam_point

    ee2base = np.eye(4)
    ee2base[:3, :3] = cv2.Rodrigues(robot_pose[3:])[0]
    ee2base[:3, 3] = robot_pose[:3]
    cam2base = ee2base @ cam2ee

    world_point = cam2base @ point_h
    return world_point[:3]

def get_euler_angles_from_rvec(rvec: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector to Euler angles (in degrees).
    
    Args:
        rvec: Rotation vector [rx, ry, rz]
    
    Returns:
        Euler angles [roll, pitch, yaw] in degrees
    """
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
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


def create_transformation_matrices(
    robot_pose: np.ndarray, 
    extrinsics: Extrinsics
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create transformation matrices from robot pose and camera extrinsics.
    
    Args:
        robot_pose: Robot pose [x, y, z, rx, ry, rz]
        extrinsics: Camera extrinsics
        
    Returns:
        Tuple of camera-to-world and world-to-camera transformation matrices
    """
    robot_pose = np.array(robot_pose)
    
    # Set up robot transformation
    robot_T = np.eye(4)
    robot_T[:3, :3] = cv2.Rodrigues(robot_pose[3:])[0]
    robot_T[:3, 3] = robot_pose[:3]
    
    # Create the camera-to-world transformation matrix
    cam2world = robot_T @ extrinsics.data
    
    # Create the world-to-camera transformation matrix (inverse of cam2world)
    world2cam = np.linalg.inv(cam2world)
    
    return cam2world, world2cam


def camera_to_world_point(
    point_camera: np.ndarray, 
    cam2world_matrix: np.ndarray
) -> np.ndarray:
    """
    Convert 3D camera coordinates to 3D world coordinates.
    
    Args:
        point_camera: Point in camera coordinates [X, Y, Z]
        cam2world_matrix: 4x4 camera-to-world transformation matrix
        
    Returns:
        3D point in world coordinates [X, Y, Z]
    """
    point_camera_h = np.append(point_camera, 1)
    point_world_h = cam2world_matrix @ point_camera_h
    return point_world_h[:3]


def world_to_camera_point(
    point_world: np.ndarray, 
    world2cam_matrix: np.ndarray
) -> np.ndarray:
    """
    Convert 3D world coordinates to 3D camera coordinates.
    
    Args:
        point_world: Point in world coordinates [X, Y, Z]
        world2cam_matrix: 4x4 world-to-camera transformation matrix
        
    Returns:
        3D point in camera coordinates [X, Y, Z]
    """
    point_world_h = np.append(point_world, 1)
    point_camera_h = world2cam_matrix @ point_world_h
    return point_camera_h[:3]


def camera_to_pixel(
    point_camera: np.ndarray, 
    intrinsics: Intrinsics
) -> np.ndarray:
    """
    Project 3D camera coordinates to pixel coordinates.
    
    Args:
        point_camera: Point in camera coordinates [X, Y, Z]
        intrinsics: Camera intrinsics
        
    Returns:
        Pixel coordinates as [row, col]
    """
    x = (point_camera[0] * intrinsics.fx / point_camera[2]) + intrinsics.cx
    y = (point_camera[1] * intrinsics.fy / point_camera[2]) + intrinsics.cy
    return np.array([int(y), int(x)])  # Return as [row, col] for image indexing


def pose_to_robot_base_frame(pose: np.ndarray, robot_pose: List[float], eye_in_hand: np.ndarray):
    # To fix Anygrasp's coordinate system
    pose_align_T = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    aligned_pose = pose @ pose_align_T

    robot_pose = np.array(robot_pose)

    robot_T = np.eye(4)
    robot_T[:3, :3] = cv2.Rodrigues(np.array(robot_pose[3:]))[0]
    robot_T[:3, 3] = robot_pose[:3]

    grasp_world_T = robot_T @ eye_in_hand @ aligned_pose

    grasp_world = np.zeros(6)
    grasp_world[:3] = grasp_world_T[:3, 3].flatten()
    grasp_world[3:] = cv2.Rodrigues(grasp_world_T[:3, :3])[0].flatten()

    return grasp_world

def get_restricted_rotation_vector(rotvec):
    """
    Given a rotation vector, find the closest vector of the form:
        v(phi) = pi * [sin(phi), -cos(phi), 0]
    Returns:
        - phi: angle in radians
        - matched_vector: the closest rotation vector in that family
        - error: Euclidean distance from input to matched vector
    """
    r = np.asarray(rotvec)
    mag = np.linalg.norm(r)
    
    # If magnitude isn't close to pi, scale to π for comparison
    if not np.isclose(mag, np.pi, atol=1e-3):
        r = (np.pi / mag) * r

    # Project onto XY plane
    x, y = r[0], r[1]
    z = r[2]

    if abs(z) > 1e-3:
        print("Warning: rotation vector not in XY plane (z ≠ 0)")

    # Recover phi from x = pi * sin(phi), y = -pi * cos(phi)
    # => sin(phi) = x/pi, cos(phi) = -y/pi
    sin_phi = x / np.pi
    cos_phi = -y / np.pi
    phi = np.arctan2(sin_phi, cos_phi)

    # Normalize angle to [0, 2π)
    phi = phi % (2 * np.pi)

    # Compute matched vector
    matched_vector = np.pi * np.array([np.sin(phi), -np.cos(phi), 0])

    return matched_vector