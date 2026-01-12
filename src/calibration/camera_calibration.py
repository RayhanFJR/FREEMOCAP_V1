"""
Modul kalibrasi kamera untuk webcam dan RealSense D435i menggunakan ChArUco board
"""
import cv2
import numpy as np
import json
import os
from typing import Tuple, Optional, List


class CameraCalibrator:
    """Kelas untuk kalibrasi kamera menggunakan ChArUco board"""
    
    def __init__(self, 
                 squares_x: int = 7,
                 squares_y: int = 5,
                 square_length: float = 40.0,
                 marker_length: float = 30.0,
                 dictionary: int = cv2.aruco.DICT_6X6_250):
        """
        Args:
            squares_x: Jumlah kotak di arah X (columns)
            squares_y: Jumlah kotak di arah Y (rows)
            square_length: Panjang sisi kotak dalam mm
            marker_length: Panjang sisi marker ArUco dalam mm
            dictionary: Dictionary ArUco yang digunakan (default: DICT_6X6_250)
        """
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            self.dictionary
        )
        
        # Detector parameters (seperti di contohcalib.py)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Arrays untuk menyimpan data kalibrasi
        self.all_charuco_corners = []
        self.all_charuco_ids = []
        self.images = []
        
    def find_charuco_board(self, img: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Mencari ChArUco board di image (mengikuti contohcalib.py)
        
        Returns:
            (found, charuco_corners, charuco_ids)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Detect ArUco markers dengan detector parameters
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, 
            self.dictionary,
            parameters=self.detector_params
        )
        
        # If at least one marker detected
        if ids is not None and len(ids) > 0:
            # Interpolate ChArUco corners
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.charuco_board
            )
            
            if ret > 0 and charuco_corners is not None and charuco_ids is not None:
                return True, charuco_corners, charuco_ids
        
        return False, None, None
    
    def add_calibration_image(self, img: np.ndarray) -> bool:
        """
        Menambahkan image untuk kalibrasi
        
        Returns:
            True jika ChArUco board ditemukan
        """
        found, charuco_corners, charuco_ids = self.find_charuco_board(img)
        if found and charuco_corners is not None and charuco_ids is not None:
            self.all_charuco_corners.append(charuco_corners)
            self.all_charuco_ids.append(charuco_ids)
            self.images.append(img.copy())
        return found
    
    def calibrate(self, img_size: Tuple[int, int]) -> Tuple[float, np.ndarray, np.ndarray, List, List]:
        """
        Melakukan kalibrasi kamera menggunakan ChArUco
        
        Args:
            img_size: Ukuran image (width, height)
            
        Returns:
            (ret, camera_matrix, dist_coeffs, rvecs, tvecs)
        """
        if len(self.all_charuco_corners) < 10:
            raise ValueError(f"Minimal 10 gambar kalibrasi diperlukan. Saat ini: {len(self.all_charuco_corners)}")
        
        # Filter out frames with too few corners (minimum 4 seperti di contohcalib.py)
        min_corners = 4
        filtered_corners = []
        filtered_ids = []
        
        for corners, ids in zip(self.all_charuco_corners, self.all_charuco_ids):
            if len(ids) >= min_corners:
                filtered_corners.append(corners)
                filtered_ids.append(ids)
        
        if len(filtered_corners) < 10:
            raise ValueError(f"Minimal 10 gambar valid diperlukan setelah filtering. Saat ini: {len(filtered_corners)}")
        
        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            filtered_corners,
            filtered_ids,
            self.charuco_board,
            img_size,
            None, None
        )
        
        return ret, mtx, dist, rvecs, tvecs
    
    def save_calibration(self, filepath: str, camera_matrix: np.ndarray, 
                        dist_coeffs: np.ndarray, img_size: Tuple[int, int]):
        """Menyimpan hasil kalibrasi ke file JSON"""
        data = {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.tolist(),
            'image_size': img_size,
            'squares_x': self.squares_x,
            'squares_y': self.squares_y,
            'square_length': self.square_length,
            'marker_length': self.marker_length,
            'type': 'charuco'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    
    def load_calibration(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Memuat hasil kalibrasi dari file JSON"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File kalibrasi tidak ditemukan: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['dist_coeffs'])
        img_size = tuple(data['image_size'])
        
        return camera_matrix, dist_coeffs, img_size
    
    def reset(self):
        """Reset semua data kalibrasi"""
        self.all_charuco_corners = []
        self.all_charuco_ids = []
        self.images = []
    
    def draw_charuco_board(self, img: np.ndarray) -> np.ndarray:
        """
        Draw ChArUco board detection pada image dengan debug info
        
        Returns:
            Image dengan deteksi ditandai
        """
        img_copy = img.copy()
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY) if len(img_copy.shape) == 3 else img_copy
        
        # Detect ArUco markers dengan detector parameters
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, 
            self.dictionary,
            parameters=self.detector_params
        )
        
        # Draw semua markers yang terdeteksi
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(img_copy, corners, ids)
            # Draw text dengan jumlah markers
            cv2.putText(img_copy, f"ArUco markers: {len(ids)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(img_copy, "No ArUco markers detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Interpolate ChArUco corners
        if ids is not None and len(ids) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.charuco_board
            )
            
            if ret > 0 and charuco_corners is not None and charuco_ids is not None:
                # Draw detected ChArUco corners
                cv2.aruco.drawDetectedCornersCharuco(
                    img_copy,
                    charuco_corners,
                    charuco_ids,
                    (255, 0, 0)  # Red color
                )
                
                # Draw text dengan jumlah corners
                cv2.putText(img_copy, f"ChArUco corners: {len(charuco_ids)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Try to estimate pose
                try:
                    ret_pose, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners,
                        charuco_ids,
                        self.charuco_board,
                        np.eye(3),
                        np.zeros((4, 1))
                    )
                    
                    if ret_pose:
                        # Draw axis (length = square_length * 2)
                        axis_length = self.square_length * 2
                        img_copy = cv2.drawFrameAxes(
                            img_copy,
                            np.eye(3),
                            np.zeros((4, 1)),
                            rvec,
                            tvec,
                            axis_length
                        )
                        cv2.putText(img_copy, "Pose OK", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except:
                    pass
            else:
                cv2.putText(img_copy, f"ChArUco interpolation failed (need more markers)", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        return img_copy


class RealSenseCalibrator(CameraCalibrator):
    """Kelas khusus untuk kalibrasi RealSense dengan depth menggunakan ChArUco"""
    
    def calibrate_depth(self, color_img: np.ndarray, depth_frame, 
                       camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Optional[np.ndarray]:
        """
        Kalibrasi depth dengan menggunakan ChArUco board
        
        Returns:
            Depth points untuk ChArUco corners (jika ditemukan)
        """
        found, charuco_corners, charuco_ids = self.find_charuco_board(color_img)
        if not found or charuco_corners is None:
            return None
        
        # Undistort corners
        h, w = color_img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        # Get depth untuk setiap corner
        depth_points = []
        for corner in charuco_corners:
            x, y = int(corner[0][0]), int(corner[0][1])
            if 0 <= x < w and 0 <= y < h:
                depth = depth_frame.get_distance(x, y)
                if depth > 0:
                    depth_points.append([corner[0][0], corner[0][1], depth])
        
        return np.array(depth_points) if depth_points else None
