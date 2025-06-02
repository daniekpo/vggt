import cv2
import numpy as np
import os

def _calculate_frame_interval(video_fps: float, target_fps: int) -> int:
    """Calculate the interval between frames to achieve target fps."""
    if target_fps <= 0:
        raise ValueError("Target fps must be positive")

    if video_fps <= 0:
        raise ValueError("Video fps must be positive")

    # If target fps is higher than video fps, take every frame
    if target_fps >= video_fps:
        return 1

    # Calculate how many frames to skip to achieve target fps
    return int(video_fps / target_fps)

def _extract_frames_from_video(cap: cv2.VideoCapture, frame_interval: int) -> list[np.ndarray]:
    """Extract frames from video capture at specified interval."""
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    return frames

def get_video_frames(video_path: str, fps: int = 30) -> list[np.ndarray]:
    """Extract frames from video at specified fps rate."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = _calculate_frame_interval(video_fps, fps)
        frames = _extract_frames_from_video(cap, frame_interval)
        return frames
    finally:
        cap.release()


def get_and_save_video_frames(video_path: str, save_dir: str, fps: int = 30):
    frames = get_video_frames(video_path, fps)
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(save_dir, f'frame_{i}.png'), frame)