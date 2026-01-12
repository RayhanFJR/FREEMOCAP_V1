"""
Modul untuk triangulasi multi-view untuk mendapatkan 3D coordinates
Terinspirasi dari FreeMoCap approach
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    rs = None


class Triangulator:
    """Kelas untuk triangulasi multi-view menggunakan stereo calibration"""
    
    def __init__(self,
                 camera_matrix1: np.ndarray,
                 dist_coeffs1: np.ndarray,
                 camera_matrix2: np.ndarray,
                 dist_coeffs2: np.ndarray,
                 R: np.ndarray,
                 T: np.ndarray):
        """
        Args:
            camera_matrix1: Camera matrix kamera 1 (webcam)
            dist_coeffs1: Distortion coefficients kamera 1
            camera_matrix2: Camera matrix kamera 2 (RealSense)
            dist_coeffs2: Distortion coefficients kamera 2
            R: Rotation matrix dari kamera 1 ke kamera 2
            T: Translation vector dari kamera 1 ke kamera 2
        """
        self.camera_matrix1 = camera_matrix1
        self.dist_coeffs1 = dist_coeffs1
        self.camera_matrix2 = camera_matrix2
        self.dist_coeffs2 = dist_coeffs2
        self.R = R
        self.T = T
        
        # Compute projection matrices
        self.P1 = self.camera_matrix1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        R2 = self.R
        t2 = self.T.reshape(3, 1)
        self.P2 = self.camera_matrix2 @ np.hstack([R2, t2])
        
        # For RealSense depth fallback
        self.rs_intrinsics = None
    
    def set_realsense_intrinsics(self, intrinsics: rs.intrinsics):
        """Set RealSense intrinsics untuk fallback ke depth-based 3D"""
        self.rs_intrinsics = intrinsics
    
    def triangulate_points(self, 
                          points_2d_1: np.ndarray,
                          points_2d_2: np.ndarray) -> np.ndarray:
        """
        Triangulasi points 2D dari dua kamera ke 3D
        
        Args:
            points_2d_1: Points 2D dari kamera 1 (N, 2)
            points_2d_2: Points 2D dari kamera 2 (N, 2)
            
        Returns:
            Points 3D (N, 3) dalam coordinate system kamera 1
        """
        if points_2d_1.shape[0] != points_2d_2.shape[0]:
            raise ValueError("Jumlah points harus sama di kedua kamera")
        
        # Undistort points
        points_2d_1_undist = cv2.undistortPoints(
            points_2d_1.reshape(-1, 1, 2),
            self.camera_matrix1,
            self.dist_coeffs1,
            P=self.camera_matrix1
        )
        points_2d_2_undist = cv2.undistortPoints(
            points_2d_2.reshape(-1, 1, 2),
            self.camera_matrix2,
            self.dist_coeffs2,
            P=self.camera_matrix2
        )
        
        # Triangulate
        points_4d = cv2.triangulatePoints(
            self.P1,
            self.P2,
            points_2d_1_undist.reshape(-1, 2).T,
            points_2d_2_undist.reshape(-1, 2).T
        )
        
        # Convert from homogeneous to 3D
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T
        
        return points_3d
    
    def triangulate_landmarks(self,
                             landmarks_2d_1: np.ndarray,
                             landmarks_2d_2: np.ndarray,
                             confidence_threshold: float = 0.5,
                             depth_frame: Optional[rs.depth_frame] = None,
                             use_depth_fallback: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Triangulasi landmarks dari dua kamera
        
        Args:
            landmarks_2d_1: Landmarks 2D dari kamera 1 (N, 2)
            landmarks_2d_2: Landmarks 2D dari kamera 2 (N, 2)
            confidence_threshold: Threshold untuk confidence (jika ada)
            depth_frame: Depth frame dari RealSense untuk fallback
            use_depth_fallback: Gunakan depth sebagai fallback jika triangulasi gagal
            
        Returns:
            (landmarks_3d, confidence_mask) - landmarks 3D dan mask untuk valid points
        """
        if landmarks_2d_1.shape[0] != landmarks_2d_2.shape[0]:
            # Pad dengan zeros jika berbeda
            max_len = max(landmarks_2d_1.shape[0], landmarks_2d_2.shape[0])
            if landmarks_2d_1.shape[0] < max_len:
                pad = np.zeros((max_len - landmarks_2d_1.shape[0], 2))
                landmarks_2d_1 = np.vstack([landmarks_2d_1, pad])
            if landmarks_2d_2.shape[0] < max_len:
                pad = np.zeros((max_len - landmarks_2d_2.shape[0], 2))
                landmarks_2d_2 = np.vstack([landmarks_2d_2, pad])
        
        # Filter invalid points (zero or negative coordinates)
        valid_mask = (landmarks_2d_1[:, 0] > 0) & (landmarks_2d_1[:, 1] > 0) & \
                     (landmarks_2d_2[:, 0] > 0) & (landmarks_2d_2[:, 1] > 0)
        
        # Initialize 3D points
        num_landmarks = landmarks_2d_1.shape[0]
        landmarks_3d = np.zeros((num_landmarks, 3))
        
        # Triangulate valid points
        if np.any(valid_mask):
            valid_points_1 = landmarks_2d_1[valid_mask]
            valid_points_2 = landmarks_2d_2[valid_mask]
            
            try:
                triangulated_3d = self.triangulate_points(valid_points_1, valid_points_2)
                landmarks_3d[valid_mask] = triangulated_3d
            except Exception as e:
                print(f"Triangulation error: {e}")
                valid_mask = np.zeros(num_landmarks, dtype=bool)
        
        # Fallback to depth-based 3D for invalid points
        if use_depth_fallback and depth_frame is not None and self.rs_intrinsics is not None:
            invalid_mask = ~valid_mask
            if np.any(invalid_mask) and landmarks_2d_2.shape[0] > 0:
                for i in np.where(invalid_mask)[0]:
                    if i < landmarks_2d_2.shape[0]:
                        x, y = int(landmarks_2d_2[i, 0]), int(landmarks_2d_2[i, 1])
                        h, w = depth_frame.get_height(), depth_frame.get_width()
                        if 0 <= x < w and 0 <= y < h:
                            depth = depth_frame.get_distance(x, y)
                            if depth > 0:
                                point_3d = rs.rs2_deproject_pixel_to_point(
                                    self.rs_intrinsics,
                                    [x, y],
                                    depth
                                )
                                # Transform from RealSense coordinate to camera 1 coordinate
                                point_3d_rs = np.array(point_3d)
                                point_3d_cam1 = self.R.T @ (point_3d_rs - self.T.reshape(3))
                                landmarks_3d[i] = point_3d_cam1
                                valid_mask[i] = True
        
        return landmarks_3d, valid_mask
    
    def reproject_to_camera(self,
                           points_3d: np.ndarray,
                           camera_id: int = 1) -> np.ndarray:
        """
        Reproject points 3D ke kamera
        
        Args:
            points_3d: Points 3D (N, 3) dalam coordinate system kamera 1
            camera_id: 1 untuk kamera 1, 2 untuk kamera 2
            
        Returns:
            Points 2D (N, 2)
        """
        if camera_id == 1:
            R_proj = np.eye(3)
            t_proj = np.zeros((3, 1))
            camera_matrix = self.camera_matrix1
            dist_coeffs = self.dist_coeffs1
        else:
            R_proj = self.R
            t_proj = self.T.reshape(3, 1)
            camera_matrix = self.camera_matrix2
            dist_coeffs = self.dist_coeffs2
        
        # Transform points
        points_3d_transformed = (R_proj @ points_3d.T + t_proj).T
        
        # Project to 2D
        points_2d, _ = cv2.projectPoints(
            points_3d_transformed.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            dist_coeffs
        )
        
        return points_2d.reshape(-1, 2)

