"""
Modul untuk filtering landmarks menggunakan Kalman Filter
Mengurangi noise dan membuat gerakan lebih smooth
"""
import numpy as np
from typing import Optional, Dict
import cv2


class KalmanFilter2D:
    """Kalman Filter untuk 2D points (x, y)"""
    
    def __init__(self, process_noise: float = 0.03, measurement_noise: float = 0.3):
        """
        Args:
            process_noise: Process noise covariance (Q) - seberapa cepat state bisa berubah
            measurement_noise: Measurement noise covariance (R) - seberapa noisy measurement
        """
        # State: [x, y, vx, vy] (position dan velocity)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Transition matrix (state transition)
        # x' = x + vx*dt, y' = y + vy*dt, vx' = vx, vy' = vy
        dt = 1.0  # Assume 1 frame time step
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (H) - kita hanya observe position
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance (Q) - noise dalam model
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance (R) - noise dalam measurement
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Error covariance (P) - initial uncertainty
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        
        # Initial state
        self.kf.statePre = np.zeros((4, 1), dtype=np.float32)
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)
        
        self.initialized = False
    
    def predict(self) -> np.ndarray:
        """Predict next state"""
        return self.kf.predict()
    
    def correct(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update dengan measurement baru
        
        Args:
            measurement: [x, y] measurement
        
        Returns:
            Corrected state [x, y, vx, vy]
        """
        if not self.initialized:
            # Initialize dengan measurement pertama
            self.kf.statePre[0] = measurement[0]
            self.kf.statePre[1] = measurement[1]
            self.kf.statePost[0] = measurement[0]
            self.kf.statePost[1] = measurement[1]
            self.initialized = True
        
        measurement_array = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        corrected = self.kf.correct(measurement_array)
        
        return corrected
    
    def get_position(self) -> np.ndarray:
        """Get current filtered position"""
        if not self.initialized:
            return np.array([0, 0])
        return np.array([self.kf.statePost[0, 0], self.kf.statePost[1, 0]])


class KalmanFilter3D:
    """Kalman Filter untuk 3D points (x, y, z)"""
    
    def __init__(self, process_noise: float = 0.03, measurement_noise: float = 0.3):
        """
        Args:
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
        """
        # State: [x, y, z, vx, vy, vz] (position dan velocity)
        self.kf = cv2.KalmanFilter(6, 3)
        
        # Transition matrix
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix - kita hanya observe position
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        
        # Error covariance
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0
        
        # Initial state
        self.kf.statePre = np.zeros((6, 1), dtype=np.float32)
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)
        
        self.initialized = False
    
    def predict(self) -> np.ndarray:
        """Predict next state"""
        return self.kf.predict()
    
    def correct(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update dengan measurement baru
        
        Args:
            measurement: [x, y, z] measurement
        
        Returns:
            Corrected state [x, y, z, vx, vy, vz]
        """
        if not self.initialized:
            # Initialize dengan measurement pertama
            self.kf.statePre[0] = measurement[0]
            self.kf.statePre[1] = measurement[1]
            self.kf.statePre[2] = measurement[2]
            self.kf.statePost[0] = measurement[0]
            self.kf.statePost[1] = measurement[1]
            self.kf.statePost[2] = measurement[2]
            self.initialized = True
        
        measurement_array = np.array([[measurement[0]], [measurement[1]], [measurement[2]]], dtype=np.float32)
        corrected = self.kf.correct(measurement_array)
        
        return corrected
    
    def get_position(self) -> np.ndarray:
        """Get current filtered position"""
        if not self.initialized:
            return np.array([0, 0, 0])
        return np.array([
            self.kf.statePost[0, 0],
            self.kf.statePost[1, 0],
            self.kf.statePost[2, 0]
        ])


class LandmarkFilter:
    """Filter untuk semua landmarks MediaPipe menggunakan Kalman Filter"""
    
    def __init__(self, 
                 num_landmarks: int = 33,
                 enable_2d_filter: bool = True,
                 enable_3d_filter: bool = True,
                 process_noise: float = 0.03,
                 measurement_noise: float = 0.3):
        """
        Args:
            num_landmarks: Jumlah landmarks (MediaPipe pose = 33)
            enable_2d_filter: Enable filtering untuk 2D landmarks
            enable_3d_filter: Enable filtering untuk 3D landmarks
            process_noise: Process noise untuk Kalman filter
            measurement_noise: Measurement noise untuk Kalman filter
        """
        self.num_landmarks = num_landmarks
        self.enable_2d_filter = enable_2d_filter
        self.enable_3d_filter = enable_3d_filter
        
        # Initialize Kalman filters untuk setiap landmark
        self.filters_2d = [KalmanFilter2D(process_noise, measurement_noise) 
                          for _ in range(num_landmarks)] if enable_2d_filter else None
        self.filters_3d = [KalmanFilter3D(process_noise, measurement_noise) 
                          for _ in range(num_landmarks)] if enable_3d_filter else None
    
    def filter_landmarks_2d(self, landmarks_2d: np.ndarray) -> np.ndarray:
        """
        Filter 2D landmarks
        
        Args:
            landmarks_2d: Array dengan shape (N, 2) berisi [x, y] untuk setiap landmark
        
        Returns:
            Filtered landmarks dengan shape yang sama
        """
        if not self.enable_2d_filter or self.filters_2d is None:
            return landmarks_2d
        
        if len(landmarks_2d) != self.num_landmarks:
            return landmarks_2d
        
        filtered = np.zeros_like(landmarks_2d)
        
        for i in range(self.num_landmarks):
            # Predict
            self.filters_2d[i].predict()
            
            # Correct dengan measurement
            self.filters_2d[i].correct(landmarks_2d[i])
            
            # Get filtered position
            filtered[i] = self.filters_2d[i].get_position()
        
        return filtered
    
    def filter_landmarks_3d(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Filter 3D landmarks
        
        Args:
            landmarks_3d: Array dengan shape (N, 3) berisi [x, y, z] untuk setiap landmark
        
        Returns:
            Filtered landmarks dengan shape yang sama
        """
        if not self.enable_3d_filter or self.filters_3d is None:
            return landmarks_3d
        
        if len(landmarks_3d) != self.num_landmarks:
            return landmarks_3d
        
        filtered = np.zeros_like(landmarks_3d)
        
        for i in range(self.num_landmarks):
            # Skip jika landmark invalid (0, 0, 0)
            if np.allclose(landmarks_3d[i], 0):
                filtered[i] = landmarks_3d[i]
                continue
            
            # Predict
            self.filters_3d[i].predict()
            
            # Correct dengan measurement
            self.filters_3d[i].correct(landmarks_3d[i])
            
            # Get filtered position
            filtered[i] = self.filters_3d[i].get_position()
        
        return filtered
    
    def reset(self):
        """Reset semua filters"""
        if self.filters_2d:
            for f in self.filters_2d:
                f.initialized = False
                f.kf.statePre = np.zeros_like(f.kf.statePre)
                f.kf.statePost = np.zeros_like(f.kf.statePost)
        
        if self.filters_3d:
            for f in self.filters_3d:
                f.initialized = False
                f.kf.statePre = np.zeros_like(f.kf.statePre)
                f.kf.statePost = np.zeros_like(f.kf.statePost)

