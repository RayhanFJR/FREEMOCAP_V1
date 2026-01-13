"""
Modul untuk tracking trajectory ankle joint (kiri dan kanan)
Menganalisis translasi dan rotasi ankle terhadap waktu
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import os
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter


class AnkleTracker:
    """Kelas untuk tracking trajectory ankle joint, heel, dan toe"""
    
    # MediaPipe landmark indices untuk ankle dan foot
    LEFT_ANKLE_IDX = 27
    RIGHT_ANKLE_IDX = 28
    LEFT_HEEL_IDX = 29
    RIGHT_HEEL_IDX = 30
    LEFT_FOOT_INDEX_IDX = 31  # Toe
    RIGHT_FOOT_INDEX_IDX = 32  # Toe
    LEFT_KNEE_IDX = 25
    RIGHT_KNEE_IDX = 26
    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24
    
    def __init__(self, smooth_window: int = 5, smooth_polyorder: int = 2):
        """
        Args:
            smooth_window: Window size untuk smoothing (harus ganjil)
            smooth_polyorder: Polynomial order untuk Savitzky-Golay filter
        """
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self.smooth_polyorder = smooth_polyorder
    
    def load_data(self, json_file: str) -> List[Dict]:
        """Load data dari file JSON"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    
    def extract_ankle_trajectories(self, data: List[Dict]) -> Dict:
        """
        Extract trajectory ankle, heel, dan toe dari data
        
        Returns:
            Dictionary dengan:
            - left_ankle_3d, right_ankle_3d: List of [x, y, z] positions
            - left_heel_3d, right_heel_3d: List of [x, y, z] positions
            - left_toe_3d, right_toe_3d: List of [x, y, z] positions
            - timestamps: List of timestamps
            - left_ankle_rotations, right_ankle_rotations: List of rotation matrices/quaternions
        """
        left_ankle_positions = []
        right_ankle_positions = []
        left_heel_positions = []
        right_heel_positions = []
        left_toe_positions = []
        right_toe_positions = []
        timestamps = []
        
        for frame in data:
            if 'landmarks_3d' in frame and frame['landmarks_3d']:
                landmarks_3d = np.array(frame['landmarks_3d'])
                
                max_idx = max(self.LEFT_ANKLE_IDX, self.RIGHT_ANKLE_IDX, 
                            self.LEFT_HEEL_IDX, self.RIGHT_HEEL_IDX,
                            self.LEFT_FOOT_INDEX_IDX, self.RIGHT_FOOT_INDEX_IDX)
                
                if len(landmarks_3d) > max_idx:
                    left_ankle = landmarks_3d[self.LEFT_ANKLE_IDX]
                    right_ankle = landmarks_3d[self.RIGHT_ANKLE_IDX]
                    left_heel = landmarks_3d[self.LEFT_HEEL_IDX]
                    right_heel = landmarks_3d[self.RIGHT_HEEL_IDX]
                    left_toe = landmarks_3d[self.LEFT_FOOT_INDEX_IDX]
                    right_toe = landmarks_3d[self.RIGHT_FOOT_INDEX_IDX]
                    
                    left_ankle_positions.append(left_ankle)
                    right_ankle_positions.append(right_ankle)
                    left_heel_positions.append(left_heel)
                    right_heel_positions.append(right_heel)
                    left_toe_positions.append(left_toe)
                    right_toe_positions.append(right_toe)
                    timestamps.append(frame.get('timestamp', len(timestamps)))
        
        # Convert to numpy arrays
        left_ankle_array = np.array(left_ankle_positions) if left_ankle_positions else np.array([])
        right_ankle_array = np.array(right_ankle_positions) if right_ankle_positions else np.array([])
        left_heel_array = np.array(left_heel_positions) if left_heel_positions else np.array([])
        right_heel_array = np.array(right_heel_positions) if right_heel_positions else np.array([])
        left_toe_array = np.array(left_toe_positions) if left_toe_positions else np.array([])
        right_toe_array = np.array(right_toe_positions) if right_toe_positions else np.array([])
        
        # Calculate rotations
        left_rotations = self._calculate_ankle_rotations(left_ankle_array, 'left', data)
        right_rotations = self._calculate_ankle_rotations(right_ankle_array, 'right', data)
        
        # Smooth trajectories
        if len(left_ankle_array) > self.smooth_window:
            left_ankle_array = self._smooth_trajectory(left_ankle_array)
        if len(right_ankle_array) > self.smooth_window:
            right_ankle_array = self._smooth_trajectory(right_ankle_array)
        if len(left_heel_array) > self.smooth_window:
            left_heel_array = self._smooth_trajectory(left_heel_array)
        if len(right_heel_array) > self.smooth_window:
            right_heel_array = self._smooth_trajectory(right_heel_array)
        if len(left_toe_array) > self.smooth_window:
            left_toe_array = self._smooth_trajectory(left_toe_array)
        if len(right_toe_array) > self.smooth_window:
            right_toe_array = self._smooth_trajectory(right_toe_array)
        
        return {
            'left_ankle_3d': left_ankle_array.tolist() if len(left_ankle_array) > 0 else [],
            'right_ankle_3d': right_ankle_array.tolist() if len(right_ankle_array) > 0 else [],
            'left_heel_3d': left_heel_array.tolist() if len(left_heel_array) > 0 else [],
            'right_heel_3d': right_heel_array.tolist() if len(right_heel_array) > 0 else [],
            'left_toe_3d': left_toe_array.tolist() if len(left_toe_array) > 0 else [],
            'right_toe_3d': right_toe_array.tolist() if len(right_toe_array) > 0 else [],
            'timestamps': timestamps,
            'left_ankle_rotations': left_rotations,
            'right_ankle_rotations': right_rotations
        }
    
    def _calculate_ankle_rotations(self, ankle_positions: np.ndarray, side: str, data: List[Dict]) -> List[Dict]:
        """
        Calculate rotation dari ankle joint menggunakan knee-ankle vector
        
        Args:
            ankle_positions: Array of ankle positions (N, 3)
            side: 'left' or 'right'
            data: Original data untuk mendapatkan knee positions
        
        Returns:
            List of rotation dictionaries dengan:
            - rotation_matrix: 3x3 rotation matrix
            - euler_angles: [x, y, z] in degrees
            - quaternion: [w, x, y, z]
        """
        rotations = []
        
        if len(ankle_positions) == 0:
            return rotations
        
        knee_idx = self.LEFT_KNEE_IDX if side == 'left' else self.RIGHT_KNEE_IDX
        hip_idx = self.LEFT_HIP_IDX if side == 'left' else self.RIGHT_HIP_IDX
        
        for i, frame in enumerate(data):
            if i >= len(ankle_positions):
                break
            
            if 'landmarks_3d' in frame and frame['landmarks_3d']:
                landmarks_3d = np.array(frame['landmarks_3d'])
                
                if len(landmarks_3d) > max(knee_idx, hip_idx):
                    ankle_pos = ankle_positions[i]
                    knee_pos = landmarks_3d[knee_idx]
                    hip_pos = landmarks_3d[hip_idx]
                    
                    # Calculate vectors
                    # Vector dari knee ke ankle (forward direction untuk ankle)
                    knee_ankle = ankle_pos - knee_pos
                    knee_ankle = knee_ankle / (np.linalg.norm(knee_ankle) + 1e-8)
                    
                    # Vector dari hip ke knee (up direction reference)
                    hip_knee = knee_pos - hip_pos
                    hip_knee = hip_knee / (np.linalg.norm(hip_knee) + 1e-8)
                    
                    # Calculate right vector (cross product)
                    right_vec = np.cross(hip_knee, knee_ankle)
                    right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-8)
                    
                    # Recalculate up vector (perpendicular to forward and right)
                    up_vec = np.cross(knee_ankle, right_vec)
                    up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-8)
                    
                    # Build rotation matrix
                    # Column vectors: right, up, forward
                    rotation_matrix = np.column_stack([right_vec, up_vec, knee_ankle])
                    
                    # Convert to scipy Rotation
                    try:
                        rot = R.from_matrix(rotation_matrix)
                        euler = rot.as_euler('xyz', degrees=True)
                        quat = rot.as_quat()  # [x, y, z, w]
                        quat = [quat[3], quat[0], quat[1], quat[2]]  # Convert to [w, x, y, z]
                    except:
                        # Fallback to identity
                        euler = np.zeros(3)
                        quat = [1, 0, 0, 0]
                    
                    rotations.append({
                        'rotation_matrix': rotation_matrix.tolist(),
                        'euler_angles': euler.tolist(),
                        'quaternion': quat
                    })
                else:
                    # Default rotation
                    rotations.append({
                        'rotation_matrix': np.eye(3).tolist(),
                        'euler_angles': [0, 0, 0],
                        'quaternion': [1, 0, 0, 0]
                    })
            else:
                # Default rotation
                rotations.append({
                    'rotation_matrix': np.eye(3).tolist(),
                    'euler_angles': [0, 0, 0],
                    'quaternion': [1, 0, 0, 0]
                })
        
        return rotations
    
    def _smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Smooth trajectory menggunakan Savitzky-Golay filter"""
        if len(trajectory) < self.smooth_window:
            return trajectory
        
        try:
            smoothed = np.zeros_like(trajectory)
            for i in range(3):  # For each x, y, z
                smoothed[:, i] = savgol_filter(trajectory[:, i], self.smooth_window, self.smooth_polyorder)
            return smoothed
        except:
            return trajectory
    
    def calculate_displacement(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Calculate displacement dari initial position
        
        Args:
            trajectory: Array of positions (N, 3)
            
        Returns:
            Displacement array (N, 3) - displacement dari initial position
        """
        if len(trajectory) == 0:
            return np.array([])
        
        initial_pos = trajectory[0]
        displacement = trajectory - initial_pos
        return displacement
    
    def calculate_velocity(self, trajectory: np.ndarray, timestamps: List[float]) -> np.ndarray:
        """
        Calculate velocity dari trajectory
        
        Args:
            trajectory: Array of positions (N, 3)
            timestamps: List of timestamps
            
        Returns:
            Velocity array (N, 3) in m/s
        """
        if len(trajectory) < 2 or len(timestamps) < 2:
            return np.zeros_like(trajectory)
        
        velocity = np.zeros_like(trajectory)
        
        for i in range(1, len(trajectory)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                velocity[i] = (trajectory[i] - trajectory[i-1]) / dt
            else:
                velocity[i] = velocity[i-1] if i > 1 else np.zeros(3)
        
        return velocity
    
    def calculate_acceleration(self, velocity: np.ndarray, timestamps: List[float]) -> np.ndarray:
        """
        Calculate acceleration dari velocity
        
        Args:
            velocity: Array of velocities (N, 3)
            timestamps: List of timestamps
            
        Returns:
            Acceleration array (N, 3) in m/s²
        """
        if len(velocity) < 2 or len(timestamps) < 2:
            return np.zeros_like(velocity)
        
        acceleration = np.zeros_like(velocity)
        
        for i in range(1, len(velocity)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                acceleration[i] = (velocity[i] - velocity[i-1]) / dt
            else:
                acceleration[i] = acceleration[i-1] if i > 1 else np.zeros(3)
        
        return acceleration
    
    def get_trajectory_stats(self, trajectory: np.ndarray, timestamps: List[float]) -> Dict:
        """
        Calculate statistics untuk trajectory
        
        Returns:
            Dictionary dengan stats: min, max, mean, std, range, total_distance
        """
        if len(trajectory) == 0:
            return {}
        
        stats = {
            'min': np.min(trajectory, axis=0).tolist(),
            'max': np.max(trajectory, axis=0).tolist(),
            'mean': np.mean(trajectory, axis=0).tolist(),
            'std': np.std(trajectory, axis=0).tolist(),
            'range': (np.max(trajectory, axis=0) - np.min(trajectory, axis=0)).tolist()
        }
        
        # Calculate total distance traveled
        total_distance = 0.0
        for i in range(1, len(trajectory)):
            total_distance += np.linalg.norm(trajectory[i] - trajectory[i-1])
        stats['total_distance'] = total_distance
        
        # Calculate duration
        if len(timestamps) > 1:
            stats['duration'] = timestamps[-1] - timestamps[0]
        else:
            stats['duration'] = 0.0
        
        return stats
    
    def calculate_ankle_angle(self, ankle_pos: np.ndarray, heel_pos: np.ndarray, 
                             toe_pos: np.ndarray) -> Dict:
        """
        Calculate ankle angle (dorsiflexion/plantarflexion, inversion/eversion)
        
        Args:
            ankle_pos: Ankle position [x, y, z]
            heel_pos: Heel position [x, y, z]
            toe_pos: Toe position [x, y, z]
            
        Returns:
            Dictionary dengan:
            - dorsiflexion_angle: Angle untuk dorsiflexion/plantarflexion (degrees)
            - inversion_angle: Angle untuk inversion/eversion (degrees)
            - foot_vector: Vector dari heel ke toe
        """
        # Vector dari heel ke toe (foot direction)
        foot_vector = toe_pos - heel_pos
        foot_length = np.linalg.norm(foot_vector)
        
        if foot_length < 1e-8:
            return {
                'dorsiflexion_angle': 0.0,
                'inversion_angle': 0.0,
                'foot_vector': [0, 0, 0]
            }
        
        foot_vector = foot_vector / foot_length
        
        # Vector dari ankle ke heel
        ankle_heel = heel_pos - ankle_pos
        ankle_heel_length = np.linalg.norm(ankle_heel)
        
        if ankle_heel_length < 1e-8:
            return {
                'dorsiflexion_angle': 0.0,
                'inversion_angle': 0.0,
                'foot_vector': foot_vector.tolist()
            }
        
        ankle_heel = ankle_heel / ankle_heel_length
        
        # Calculate dorsiflexion/plantarflexion angle
        # Angle antara foot vector dan vertical (Y-axis)
        vertical = np.array([0, 1, 0])  # Up direction
        dorsiflexion_angle = np.degrees(np.arccos(np.clip(np.dot(foot_vector, vertical), -1, 1)))
        
        # Calculate inversion/eversion angle
        # Project foot vector ke horizontal plane (XZ plane)
        foot_horizontal = np.array([foot_vector[0], 0, foot_vector[2]])
        foot_horizontal = foot_horizontal / (np.linalg.norm(foot_horizontal) + 1e-8)
        
        # Angle dari forward direction (Z-axis)
        forward = np.array([0, 0, 1])
        inversion_angle = np.degrees(np.arccos(np.clip(np.dot(foot_horizontal, forward), -1, 1)))
        
        return {
            'dorsiflexion_angle': dorsiflexion_angle,
            'inversion_angle': inversion_angle,
            'foot_vector': foot_vector.tolist()
        }
    
    def analyze_motion_trials(self, trajectory_data: Dict) -> Dict:
        """
        Analisis motion trials: dorsiflexion/plantarflexion, inversion/eversion, circular gait
        
        Returns:
            Dictionary dengan analisis untuk setiap motion type
        """
        results = {
            'dorsiflexion_plantarflexion': {},
            'inversion_eversion': {},
            'circular_gait': {}
        }
        
        # Extract data
        left_ankle = np.array(trajectory_data.get('left_ankle_3d', []))
        right_ankle = np.array(trajectory_data.get('right_ankle_3d', []))
        left_heel = np.array(trajectory_data.get('left_heel_3d', []))
        right_heel = np.array(trajectory_data.get('right_heel_3d', []))
        left_toe = np.array(trajectory_data.get('left_toe_3d', []))
        right_toe = np.array(trajectory_data.get('right_toe_3d', []))
        
        # Analyze left foot
        if len(left_ankle) > 0 and len(left_heel) > 0 and len(left_toe) > 0:
            left_angles = []
            for i in range(len(left_ankle)):
                angle_data = self.calculate_ankle_angle(left_ankle[i], left_heel[i], left_toe[i])
                left_angles.append(angle_data)
            
            results['dorsiflexion_plantarflexion']['left'] = {
                'angles': [a['dorsiflexion_angle'] for a in left_angles],
                'min': min([a['dorsiflexion_angle'] for a in left_angles]) if left_angles else 0,
                'max': max([a['dorsiflexion_angle'] for a in left_angles]) if left_angles else 0,
                'range': max([a['dorsiflexion_angle'] for a in left_angles]) - min([a['dorsiflexion_angle'] for a in left_angles]) if left_angles else 0
            }
            
            results['inversion_eversion']['left'] = {
                'angles': [a['inversion_angle'] for a in left_angles],
                'min': min([a['inversion_angle'] for a in left_angles]) if left_angles else 0,
                'max': max([a['inversion_angle'] for a in left_angles]) if left_angles else 0,
                'range': max([a['inversion_angle'] for a in left_angles]) - min([a['inversion_angle'] for a in left_angles]) if left_angles else 0
            }
        
        # Analyze right foot
        if len(right_ankle) > 0 and len(right_heel) > 0 and len(right_toe) > 0:
            right_angles = []
            for i in range(len(right_ankle)):
                angle_data = self.calculate_ankle_angle(right_ankle[i], right_heel[i], right_toe[i])
                right_angles.append(angle_data)
            
            results['dorsiflexion_plantarflexion']['right'] = {
                'angles': [a['dorsiflexion_angle'] for a in right_angles],
                'min': min([a['dorsiflexion_angle'] for a in right_angles]) if right_angles else 0,
                'max': max([a['dorsiflexion_angle'] for a in right_angles]) if right_angles else 0,
                'range': max([a['dorsiflexion_angle'] for a in right_angles]) - min([a['dorsiflexion_angle'] for a in right_angles]) if right_angles else 0
            }
            
            results['inversion_eversion']['right'] = {
                'angles': [a['inversion_angle'] for a in right_angles],
                'min': min([a['inversion_angle'] for a in right_angles]) if right_angles else 0,
                'max': max([a['inversion_angle'] for a in right_angles]) if right_angles else 0,
                'range': max([a['inversion_angle'] for a in right_angles]) - min([a['inversion_angle'] for a in right_angles]) if right_angles else 0
            }
        
        # Analyze circular gait (trajectory dalam horizontal plane)
        if len(left_ankle) > 0:
            left_ankle_2d = left_ankle[:, [0, 2]]  # X and Z (horizontal plane)
            # Calculate if trajectory is circular
            center = np.mean(left_ankle_2d, axis=0)
            distances = np.linalg.norm(left_ankle_2d - center, axis=1)
            mean_radius = np.mean(distances)
            std_radius = np.std(distances)
            circularity = 1.0 - (std_radius / (mean_radius + 1e-8))  # Closer to 1 = more circular
            
            results['circular_gait']['left'] = {
                'center': center.tolist(),
                'mean_radius': float(mean_radius),
                'std_radius': float(std_radius),
                'circularity': float(circularity)
            }
        
        if len(right_ankle) > 0:
            right_ankle_2d = right_ankle[:, [0, 2]]  # X and Z (horizontal plane)
            center = np.mean(right_ankle_2d, axis=0)
            distances = np.linalg.norm(right_ankle_2d - center, axis=1)
            mean_radius = np.mean(distances)
            std_radius = np.std(distances)
            circularity = 1.0 - (std_radius / (mean_radius + 1e-8))
            
            results['circular_gait']['right'] = {
                'center': center.tolist(),
                'mean_radius': float(mean_radius),
                'std_radius': float(std_radius),
                'circularity': float(circularity)
            }
        
        return results
    
    def export_trajectory(self, trajectory_data: Dict, output_file: str):
        """Export trajectory data ke JSON"""
        with open(output_file, 'w') as f:
            json.dump(trajectory_data, f, indent=2)

