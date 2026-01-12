"""
Modul pose estimation menggunakan MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple, Dict, Any
import json


class PoseEstimator:
    """Kelas untuk pose estimation menggunakan MediaPipe"""
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 enable_segmentation: bool = False):
        """
        Args:
            model_complexity: 0, 1, or 2 (higher = more accurate but slower)
            min_detection_confidence: Minimum confidence untuk detection
            min_tracking_confidence: Minimum confidence untuk tracking
            enable_segmentation: Enable body segmentation
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=enable_segmentation
        )
        
        # Landmark names untuk referensi
        self.landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear',
            'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
            'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
            'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
            'right_index', 'left_thumb', 'right_thumb', 'left_hip',
            'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def process(self, image: np.ndarray) -> Optional[Any]:
        """
        Process image untuk detect pose
        
        Returns:
            Pose landmarks atau None jika tidak ada pose terdeteksi
        """
        if image is None:
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process image
        results = self.pose.process(image_rgb)
        
        return results.pose_landmarks if results.pose_landmarks else None
    
    def draw_landmarks(self, image: np.ndarray, landmarks: Any) -> np.ndarray:
        """Draw pose landmarks pada image"""
        if landmarks is None:
            return image
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = True
        
        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            image_rgb,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Convert back to BGR
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    def get_landmarks_2d(self, landmarks: Any, 
                        image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert landmarks ke 2D pixel coordinates
        
        Returns:
            Array dengan shape (33, 2) berisi [x, y] untuk setiap landmark
        """
        if landmarks is None:
            return np.array([])
        
        h, w = image_shape[:2]
        points = []
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        
        return np.array(points)
    
    def get_landmarks_3d(self, landmarks: Any,
                        depth_frame, intrinsics) -> Optional[np.ndarray]:
        """
        Convert landmarks ke 3D coordinates menggunakan depth data
        
        Returns:
            Array dengan shape (33, 3) berisi [x, y, z] untuk setiap landmark
        """
        if landmarks is None or depth_frame is None:
            return None
        
        import pyrealsense2 as rs
        
        h, w = depth_frame.get_height(), depth_frame.get_width()
        points_3d = []
        
        for landmark in landmarks.landmark:
            x_pixel = int(landmark.x * w)
            y_pixel = int(landmark.y * h)
            
            if 0 <= x_pixel < w and 0 <= y_pixel < h:
                depth = depth_frame.get_distance(x_pixel, y_pixel)
                
                if depth > 0:
                    # Convert pixel + depth to 3D point
                    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x_pixel, y_pixel], depth)
                    points_3d.append(point_3d)
                else:
                    points_3d.append([0, 0, 0])
            else:
                points_3d.append([0, 0, 0])
        
        return np.array(points_3d)
    
    def landmarks_to_dict(self, landmarks: Any,
                         image_shape: Tuple[int, int],
                         depth_frame=None,
                         intrinsics=None) -> Dict:
        """
        Convert landmarks ke dictionary format untuk export
        
        Returns:
            Dictionary dengan data landmarks
        """
        if landmarks is None:
            return {}
        
        data = {
            'landmarks_2d': self.get_landmarks_2d(landmarks, image_shape).tolist(),
            'landmark_names': self.landmark_names,
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        }
        
        if depth_frame is not None and intrinsics is not None:
            landmarks_3d = self.get_landmarks_3d(landmarks, depth_frame, intrinsics)
            if landmarks_3d is not None:
                data['landmarks_3d'] = landmarks_3d.tolist()
        
        return data
    
    def save_landmarks(self, filepath: str, landmarks_data: List[Dict]):
        """Save landmarks data ke JSON file"""
        with open(filepath, 'w') as f:
            json.dump(landmarks_data, f, indent=2)
    
    def release(self):
        """Release pose estimator"""
        if self.pose:
            self.pose.close()

