"""
Modul pose estimation menggunakan MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple, Dict, Any
import json
from .landmark_filter import LandmarkFilter
from .triangulation import Triangulator


class PoseEstimator:
    """Kelas untuk pose estimation menggunakan MediaPipe"""
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 enable_segmentation: bool = False,
                 enable_filter: bool = True,
                 filter_process_noise: float = 0.03,
                 filter_measurement_noise: float = 0.3):
        """
        Args:
            model_complexity: 0, 1, or 2 (higher = more accurate but slower)
            min_detection_confidence: Minimum confidence untuk detection
            min_tracking_confidence: Minimum confidence untuk tracking
            enable_segmentation: Enable body segmentation
            enable_filter: Enable Kalman filter untuk smoothing landmarks
            filter_process_noise: Process noise untuk Kalman filter
            filter_measurement_noise: Measurement noise untuk Kalman filter
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
        
        # Initialize landmark filter
        self.enable_filter = enable_filter
        self.landmark_filter = LandmarkFilter(
            num_landmarks=33,
            enable_2d_filter=enable_filter,
            enable_3d_filter=enable_filter,
            process_noise=filter_process_noise,
            measurement_noise=filter_measurement_noise
        ) if enable_filter else None
        
        # Triangulator untuk multi-view 3D reconstruction
        self.triangulator = None
        self.use_triangulation = False
    
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
                        image_shape: Tuple[int, int],
                        apply_filter: bool = True) -> np.ndarray:
        """
        Convert landmarks ke 2D pixel coordinates
        
        Args:
            landmarks: MediaPipe landmarks
            image_shape: (height, width) dari image
            apply_filter: Apply Kalman filter jika enabled
        
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
        
        points = np.array(points)
        
        # Apply filter jika enabled
        if apply_filter and self.enable_filter and self.landmark_filter is not None:
            points = self.landmark_filter.filter_landmarks_2d(points)
        
        return points
    
    def set_triangulator(self, triangulator: Triangulator):
        """Set triangulator untuk multi-view 3D reconstruction"""
        self.triangulator = triangulator
        self.use_triangulation = (triangulator is not None)
    
    def get_landmarks_3d(self, landmarks: Any,
                        depth_frame=None, intrinsics=None,
                        landmarks_2d_webcam: Optional[np.ndarray] = None,
                        apply_filter: bool = True) -> Optional[np.ndarray]:
        """
        Convert landmarks ke 3D coordinates menggunakan triangulasi (prioritas) atau depth data (fallback)
        
        Args:
            landmarks: MediaPipe landmarks dari RealSense
            depth_frame: RealSense depth frame (untuk fallback)
            intrinsics: RealSense camera intrinsics (untuk fallback)
            landmarks_2d_webcam: Landmarks 2D dari webcam (untuk triangulasi)
            apply_filter: Apply Kalman filter jika enabled
        
        Returns:
            Array dengan shape (33, 3) berisi [x, y, z] untuk setiap landmark
        """
        if landmarks is None:
            return None
        
        # Prioritas 1: Gunakan triangulasi jika tersedia
        if self.use_triangulation and self.triangulator is not None and landmarks_2d_webcam is not None:
            landmarks_2d_rs = self.get_landmarks_2d(landmarks, (depth_frame.get_height(), depth_frame.get_width()) if depth_frame else (480, 640), apply_filter=False)
            
            if len(landmarks_2d_rs) > 0 and len(landmarks_2d_webcam) > 0:
                try:
                    # Triangulate dengan fallback ke depth
                    points_3d, valid_mask = self.triangulator.triangulate_landmarks(
                        landmarks_2d_webcam,
                        landmarks_2d_rs,
                        depth_frame=depth_frame,
                        use_depth_fallback=True
                    )
                    
                    # Apply filter jika enabled
                    if apply_filter and self.enable_filter and self.landmark_filter is not None:
                        points_3d = self.landmark_filter.filter_landmarks_3d(points_3d)
                    
                    return points_3d
                except Exception as e:
                    print(f"Triangulation failed, falling back to depth: {e}")
                    # Fall through to depth-based method
        
        # Fallback: Gunakan depth-based 3D
        if depth_frame is None or intrinsics is None:
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
        
        points_3d = np.array(points_3d)
        
        # Apply filter jika enabled
        if apply_filter and self.enable_filter and self.landmark_filter is not None:
            points_3d = self.landmark_filter.filter_landmarks_3d(points_3d)
        
        return points_3d
    
    def landmarks_to_dict(self, landmarks: Any,
                         image_shape: Tuple[int, int],
                         depth_frame=None,
                         intrinsics=None,
                         landmarks_2d_webcam: Optional[np.ndarray] = None,
                         apply_filter: bool = True) -> Dict:
        """
        Convert landmarks ke dictionary format untuk export
        
        Args:
            landmarks: MediaPipe landmarks dari RealSense
            image_shape: (height, width) dari image
            depth_frame: RealSense depth frame (optional, untuk fallback)
            intrinsics: RealSense camera intrinsics (optional, untuk fallback)
            landmarks_2d_webcam: Landmarks 2D dari webcam (untuk triangulasi)
            apply_filter: Apply Kalman filter jika enabled
        
        Returns:
            Dictionary dengan data landmarks
        """
        if landmarks is None:
            return {}
        
        data = {
            'landmarks_2d': self.get_landmarks_2d(landmarks, image_shape, apply_filter).tolist(),
            'landmark_names': self.landmark_names,
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        }
        
        # Get 3D landmarks menggunakan triangulasi (prioritas) atau depth (fallback)
        landmarks_3d = self.get_landmarks_3d(
            landmarks, 
            depth_frame=depth_frame,
            intrinsics=intrinsics,
            landmarks_2d_webcam=landmarks_2d_webcam,
            apply_filter=apply_filter
        )
        
        if landmarks_3d is not None:
            data['landmarks_3d'] = landmarks_3d.tolist()
            # Tandai metode yang digunakan
            data['reconstruction_method'] = 'triangulation' if self.use_triangulation and landmarks_2d_webcam is not None else 'depth'
        
        return data
    
    def set_filter_enabled(self, enabled: bool):
        """Enable/disable Kalman filter"""
        self.enable_filter = enabled
        if enabled and self.landmark_filter is None:
            self.landmark_filter = LandmarkFilter(
                num_landmarks=33,
                enable_2d_filter=True,
                enable_3d_filter=True
            )
        elif not enabled:
            if self.landmark_filter is not None:
                self.landmark_filter.reset()
    
    def reset_filter(self):
        """Reset Kalman filter state"""
        if self.landmark_filter is not None:
            self.landmark_filter.reset()
    
    def save_landmarks(self, filepath: str, landmarks_data: List[Dict]):
        """Save landmarks data ke JSON file"""
        with open(filepath, 'w') as f:
            json.dump(landmarks_data, f, indent=2)
    
    def release(self):
        """Release pose estimator"""
        if self.pose:
            self.pose.close()

