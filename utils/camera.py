import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclasses.dataclass
class Extrinsics:
    data: np.ndarray

    @classmethod
    def from_json(cls, extrinsics: Union[str, np.ndarray, list]):
        """
        Create an Extrinsics object from a JSON file path or array.

        Args:
            extrinsics (Union[str, numpy.ndarray]): JSON file path or numpy array of 4x4 matrix.
        """
        if isinstance(extrinsics, str):
            import json
            with open(extrinsics) as f:
                extrinsics = np.array(json.load(f))
        elif isinstance(extrinsics, np.ndarray):
            extrinsics = extrinsics
        elif isinstance(extrinsics, list):
            extrinsics = np.array(extrinsics)
        else:
            raise ValueError("Extrinsics must be a JSON file path or a numpy array.")


        return cls(
            data=extrinsics
        )

@dataclasses.dataclass
class Intrinsics:
    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_json(cls, intrinsics: Union[str, dict]):
        """
        Create an Intrinsics object from a JSON file path or dictionary.

        Args:
            intrinsics (Union[str, dict]): JSON file path or dictionary containing intrinsics.
        """
        if isinstance(intrinsics, str):
            import json
            with open(intrinsics) as f:
                intrinsics = json.load(f)

        return cls(
            height=intrinsics["height"],
            width=intrinsics["width"],
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
        )

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


def campoint_to_worldpoint(cam_point, extrinsics: Extrinsics, robotl: np.ndarray):
    cam2ee = extrinsics.data

    point_h = np.ones(4)
    point_h[:3] = cam_point

    ee2base = np.eye(4)
    ee2base[:3, :3] = cv2.Rodrigues(robotl[3:])[0]
    ee2base[:3, 3] = robotl[:3]
    cam2base = ee2base @ cam2ee

    world_point = cam2base @ point_h
    return world_point[:3]

def cam_pose_to_world_pose(cam_pose, extrinsics: Extrinsics, robotl: Union[np.ndarray, list]):
    """
    cam_pose: np.ndarray of shape (6,)
    extrinsics: Extrinsics object
    robotl: np.ndarray of shape (6,)
    """
    obj_transform = np.eye(4)
    obj_transform[:3, :3] = cv2.Rodrigues(cam_pose[3:])[0]
    obj_transform[:3, 3] = cam_pose[:3]

    if not isinstance(robotl, np.ndarray):
        robotl = np.array(robotl)
    ee2base = np.eye(4)
    ee2base[:3, :3] = cv2.Rodrigues(robotl[3:])[0]
    ee2base[:3, 3] = robotl[:3]
    cam2base = ee2base @ extrinsics.data

    world_pose = cam2base @ obj_transform

    translation = world_pose[:3, 3]
    rotation = cv2.Rodrigues(world_pose[:3, :3])[0].flatten()
    return np.concatenate([translation, rotation])

def create_camera_matrix(intrinsics: Intrinsics) -> np.ndarray:
    """
    Create a camera matrix from intrinsics.
    
    Args:
        intrinsics: Camera intrinsics
        
    Returns:
        Camera matrix as 3x3 array
    """
    return np.array([
        [intrinsics.fx, 0, intrinsics.cx],
        [0, intrinsics.fy, intrinsics.cy],
        [0, 0, 1]
    ])

def estimate_marker_pose(
    corners: np.ndarray, 
    marker_id: int, 
    camera_matrix: np.ndarray, 
    dist_coeffs: np.ndarray = None, 
    marker_size: float = 0.015
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate pose of a single ArUco marker.
    
    Args:
        corners: Corner points of the marker
        marker_id: ID of the marker
        camera_matrix: Camera matrix
        dist_coeffs: Distortion coefficients, default is None (no distortion)
        marker_size: Size of the marker in meters
        
    Returns:
        Tuple of (rvecs, tvecs) - rotation and translation vectors
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1))
        
    # Reshape corners to format expected by estimatePoseSingleMarkers
    corners_for_pose = np.array([corners[0]], dtype=np.float32)

    # Estimate the pose
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners_for_pose, marker_size, camera_matrix, dist_coeffs
    )

    return rvecs, tvecs

def detect_aruco_markers(
    gray_img: np.ndarray, 
    aruco_dict_type: int = cv2.aruco.DICT_5X5_100,
    parameters: dict = None
) -> Tuple[List, List]:
    """
    Detect ArUco markers in an image.
    
    Args:
        gray_img: Grayscale image
        aruco_dict_type: Type of ArUco dictionary
        parameters: ArUco detection parameters
        
    Returns:
        Tuple of (corners, ids)
    """
    if parameters is None:
        parameters = cv2.aruco.DetectorParameters()
        
    # Load the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    
    # Create the detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray_img)
    
    return corners, ids

def detect_charuco_board(
    gray_img: np.ndarray,
    squares_x: int = 9,
    squares_y: int = 14, 
    square_length: float = 0.02,
    marker_length: float = 0.015,
    aruco_dict_type: int = cv2.aruco.DICT_5X5_100
) -> Tuple[np.ndarray, np.ndarray, List, List]:
    """
    Detect ChArUco board in an image.
    
    Args:
        gray_img: Grayscale image
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length: Square length in meters
        marker_length: Marker length in meters
        aruco_dict_type: Type of ArUco dictionary
        
    Returns:
        Tuple of (charuco_corners, charuco_ids, marker_corners, marker_ids)
    """
    # Create ChArUco board
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    charuco_board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, dictionary
    )
    detector = cv2.aruco.CharucoDetector(board=charuco_board)
    
    # Detect board
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray_img)
    
    return charuco_corners, charuco_ids, marker_corners, marker_ids

def get_marker_pose_from_detection(
    x: int, 
    y: int, 
    depth: np.ndarray, 
    intrinsics: Intrinsics, 
    extrinsics: Extrinsics, 
    robot_pose: np.ndarray, 
    rvecs: np.ndarray,
    fixed_orientation: bool = True
) -> np.ndarray:
    """
    Calculate the 6DOF pose of a marker from detection and convert to robot base frame.
    
    Args:
        x: X coordinate of marker corner pixel
        y: Y coordinate of marker corner pixel
        depth: Depth image (in meters)
        intrinsics: Camera intrinsics
        extrinsics: Camera extrinsics (eye-in-hand transform)
        robot_pose: Robot pose [x, y, z, rx, ry, rz]
        rvecs: Rotation vector from marker detection
        fixed_orientation: Whether to use fixed downward orientation
        
    Returns:
        6DOF pose vector [x, y, z, rx, ry, rz] in robot base frame
    """
    # Get camera point from pixel
    cam_point = pixel_to_camera((x, y), depth, intrinsics)
    
    # Create camera pose
    cam_pose = np.zeros(6)
    cam_pose[:3] = cam_point  # Position
    cam_pose[3:] = rvecs[0][0]  # Rotation
    
    # Transform camera pose to world pose
    world_pose = cam_pose_to_world_pose(cam_pose, extrinsics, robot_pose)
    
    # Use fixed downward orientation if specified
    if fixed_orientation:
        world_pose[3:] = np.array([0, -np.pi, 0])
    
    return world_pose

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
