"""
Utility functions
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import os
import json


def create_output_dir(base_dir: str = "output") -> str:
    """Create output directory jika belum ada"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir


def save_frame(frame: np.ndarray, filepath: str):
    """Save frame ke file"""
    cv2.imwrite(filepath, frame)


def combine_frames_horizontal(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Combine dua frames secara horizontal"""
    if frame1 is None or frame2 is None:
        return frame1 if frame1 is not None else frame2
    
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    # Resize ke tinggi yang sama
    if h1 != h2:
        scale = min(h1, h2) / max(h1, h2)
        if h1 > h2:
            frame1 = cv2.resize(frame1, (int(w1 * scale), h2))
            w1 = int(w1 * scale)
        else:
            frame2 = cv2.resize(frame2, (int(w2 * scale), h1))
            w2 = int(w2 * scale)
    
    return np.hstack([frame1, frame2])


def combine_frames_vertical(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Combine dua frames secara vertical"""
    if frame1 is None or frame2 is None:
        return frame1 if frame1 is not None else frame2
    
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    # Resize ke lebar yang sama
    if w1 != w2:
        scale = min(w1, w2) / max(w1, w2)
        if w1 > w2:
            frame1 = cv2.resize(frame1, (w2, int(h1 * scale)))
            h1 = int(h1 * scale)
        else:
            frame2 = cv2.resize(frame2, (w1, int(h2 * scale)))
            h2 = int(h2 * scale)
    
    return np.vstack([frame1, frame2])


def draw_text(image: np.ndarray, text: str, position: Tuple[int, int],
             font_scale: float = 0.7, color: Tuple[int, int, int] = (0, 255, 0),
             thickness: int = 2):
    """Draw text pada image"""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, color, thickness, cv2.LINE_AA)


def get_fps(start_time: float, frame_count: int) -> float:
    """Calculate FPS"""
    if frame_count == 0:
        return 0.0
    elapsed = cv2.getTickCount() / cv2.getTickFrequency() - start_time
    return frame_count / elapsed if elapsed > 0 else 0.0

