"""
Modul untuk stereo calibration antara webcam dan RealSense menggunakan ChArUco board
Terinspirasi dari FreeMoCap approach
"""
import cv2
import numpy as np
import json
import os
from typing import Tuple, Optional, List
from .camera_calibration import CameraCalibrator


class StereoCalibrator:
    """Kelas untuk stereo calibration antara dua kamera menggunakan ChArUco board"""
    
    def __init__(self, calibrator1: CameraCalibrator, calibrator2: CameraCalibrator):
        """
        Args:
            calibrator1: Calibrator untuk kamera pertama (webcam)
            calibrator2: Calibrator untuk kamera kedua (RealSense)
        """
        self.calibrator1 = calibrator1
        self.calibrator2 = calibrator2
        
        # Arrays untuk menyimpan data stereo calibration
        self.image_pairs = []  # Pairs of images from both cameras
        self.charuco_corners_1 = []
        self.charuco_ids_1 = []
        self.charuco_corners_2 = []
        self.charuco_ids_2 = []
        
    def find_charuco_in_pair(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Mencari ChArUco board di kedua gambar secara simultan
        
        Returns:
            (found, corners1, ids1, corners2, ids2)
        """
        found1, corners1, ids1 = self.calibrator1.find_charuco_board(img1)
        found2, corners2, ids2 = self.calibrator2.find_charuco_board(img2)
        
        # Hanya return True jika kedua kamera mendeteksi board
        if found1 and found2 and corners1 is not None and corners2 is not None:
            # Pastikan jumlah IDs sama (same board detected)
            if ids1 is not None and ids2 is not None:
                # Check if there are common IDs (at least some overlap)
                common_ids = set(ids1.flatten()) & set(ids2.flatten())
                if len(common_ids) >= 4:  # Minimum 4 common corners
                    return True, corners1, ids1, corners2, ids2
        
        return False, None, None, None, None
    
    def add_stereo_pair(self, img1: np.ndarray, img2: np.ndarray) -> bool:
        """
        Menambahkan pasangan gambar untuk stereo calibration
        
        Returns:
            True jika ChArUco board ditemukan di kedua gambar
        """
        found, corners1, ids1, corners2, ids2 = self.find_charuco_in_pair(img1, img2)
        
        if found and corners1 is not None and corners2 is not None:
            self.charuco_corners_1.append(corners1)
            self.charuco_ids_1.append(ids1)
            self.charuco_corners_2.append(corners2)
            self.charuco_ids_2.append(ids2)
            self.image_pairs.append((img1.copy(), img2.copy()))
            return True
        
        return False
    
    def calibrate_stereo(self, 
                        camera_matrix1: np.ndarray, dist_coeffs1: np.ndarray,
                        camera_matrix2: np.ndarray, dist_coeffs2: np.ndarray,
                        img_size: Tuple[int, int],
                        flags: int = cv2.CALIB_FIX_INTRINSIC) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Melakukan stereo calibration
        
        Args:
            camera_matrix1: Camera matrix kamera 1 (sudah dikalibrasi)
            dist_coeffs1: Distortion coefficients kamera 1
            camera_matrix2: Camera matrix kamera 2 (sudah dikalibrasi)
            dist_coeffs2: Distortion coefficients kamera 2
            img_size: Ukuran gambar (width, height)
            flags: Calibration flags
            
        Returns:
            (ret, R, T, E, F) - Rotation, Translation, Essential, Fundamental matrices
        """
        if len(self.charuco_corners_1) < 10:
            raise ValueError(f"Minimal 10 pasangan gambar diperlukan. Saat ini: {len(self.charuco_corners_1)}")
        
        # Filter out frames with too few corners
        min_corners = 4
        filtered_corners_1 = []
        filtered_ids_1 = []
        filtered_corners_2 = []
        filtered_ids_2 = []
        
        for corners1, ids1, corners2, ids2 in zip(
            self.charuco_corners_1, self.charuco_ids_1,
            self.charuco_corners_2, self.charuco_ids_2
        ):
            if len(ids1) >= min_corners and len(ids2) >= min_corners:
                # Filter untuk common IDs only
                common_ids = set(ids1.flatten()) & set(ids2.flatten())
                if len(common_ids) >= min_corners:
                    # Find indices for common IDs
                    mask1 = np.isin(ids1.flatten(), list(common_ids))
                    mask2 = np.isin(ids2.flatten(), list(common_ids))
                    
                    filtered_corners_1.append(corners1[mask1])
                    filtered_ids_1.append(ids1[mask1])
                    filtered_corners_2.append(corners2[mask2])
                    filtered_ids_2.append(ids2[mask2])
        
        if len(filtered_corners_1) < 10:
            raise ValueError(f"Minimal 10 pasangan valid diperlukan setelah filtering. Saat ini: {len(filtered_corners_1)}")
        
        # Convert ChArUco corners to object points (3D points on board)
        # We need to match corners by ID and get their 3D positions
        obj_points_list = []
        img_points_1_list = []
        img_points_2_list = []
        
        board = self.calibrator1.charuco_board
        
        for corners1, ids1, corners2, ids2 in zip(
            filtered_corners_1, filtered_ids_1,
            filtered_corners_2, filtered_ids_2
        ):
            # Get 3D object points for common IDs
            obj_points = []
            img_points_1 = []
            img_points_2 = []
            
            # Get all IDs that appear in both
            ids1_flat = ids1.flatten()
            ids2_flat = ids2.flatten()
            common_ids = set(ids1_flat) & set(ids2_flat)
            
            for id_val in common_ids:
                # Get 3D position from board
                obj_point = board.chessboardCorners[id_val]
                obj_points.append(obj_point)
                
                # Get corresponding 2D points
                idx1 = np.where(ids1_flat == id_val)[0][0]
                idx2 = np.where(ids2_flat == id_val)[0][0]
                img_points_1.append(corners1[idx1][0])
                img_points_2.append(corners2[idx2][0])
            
            if len(obj_points) >= min_corners:
                obj_points_list.append(np.array(obj_points, dtype=np.float32))
                img_points_1_list.append(np.array(img_points_1, dtype=np.float32))
                img_points_2_list.append(np.array(img_points_2, dtype=np.float32))
        
        if len(obj_points_list) < 10:
            raise ValueError(f"Minimal 10 pasangan valid diperlukan. Saat ini: {len(obj_points_list)}")
        
        # Perform stereo calibration
        ret, camera_matrix1_new, dist_coeffs1_new, camera_matrix2_new, dist_coeffs2_new, \
        R, T, E, F = cv2.stereoCalibrate(
            obj_points_list,
            img_points_1_list,
            img_points_2_list,
            camera_matrix1,
            dist_coeffs1,
            camera_matrix2,
            dist_coeffs2,
            img_size,
            flags=flags
        )
        
        return ret, R, T, E, F
    
    def save_stereo_calibration(self, filepath: str,
                                R: np.ndarray, T: np.ndarray,
                                E: np.ndarray, F: np.ndarray):
        """Menyimpan hasil stereo calibration ke file JSON"""
        data = {
            'R': R.tolist(),
            'T': T.tolist(),
            'E': E.tolist(),
            'F': F.tolist(),
            'type': 'stereo_charuco'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    
    def load_stereo_calibration(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Memuat hasil stereo calibration dari file JSON"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File stereo calibration tidak ditemukan: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        R = np.array(data['R'])
        T = np.array(data['T'])
        E = np.array(data['E'])
        F = np.array(data['F'])
        
        return R, T, E, F
    
    def reset(self):
        """Reset semua data stereo calibration"""
        self.image_pairs = []
        self.charuco_corners_1 = []
        self.charuco_ids_1 = []
        self.charuco_corners_2 = []
        self.charuco_ids_2 = []

