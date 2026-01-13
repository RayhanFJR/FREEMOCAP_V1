"""
Modul untuk visualisasi trajectory ankle joint
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import List, Dict, Optional
import sys
import os
# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tracking.ankle_tracker import AnkleTracker


class TrajectoryVisualizer:
    """Kelas untuk visualisasi trajectory ankle"""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.tracker = AnkleTracker()
        self.trajectory_data = None
        
    def load_and_analyze(self, json_file: str):
        """Load data dan analisis trajectory"""
        data = self.tracker.load_data(json_file)
        self.trajectory_data = self.tracker.extract_ankle_trajectories(data)
        return self.trajectory_data
    
    def plot_position_vs_time(self, side: str = 'both', save_path: Optional[str] = None):
        """
        Plot posisi ankle terhadap waktu untuk X, Y, Z
        
        Args:
            side: 'left', 'right', or 'both'
            save_path: Path untuk save gambar (optional)
        """
        if self.trajectory_data is None:
            print("No trajectory data loaded")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        timestamps = np.array(self.trajectory_data['timestamps'])
        if len(timestamps) > 1:
            timestamps = timestamps - timestamps[0]  # Normalize to start from 0
        
        if side in ['left', 'both'] and len(self.trajectory_data['left_ankle_3d']) > 0:
            left_ankle = np.array(self.trajectory_data['left_ankle_3d'])
            
            axes[0].plot(timestamps, left_ankle[:, 0], 'b-', label='Left Ankle X', linewidth=2)
            axes[1].plot(timestamps, left_ankle[:, 1], 'b-', label='Left Ankle Y', linewidth=2)
            axes[2].plot(timestamps, left_ankle[:, 2], 'b-', label='Left Ankle Z', linewidth=2)
        
        if side in ['right', 'both'] and len(self.trajectory_data['right_ankle_3d']) > 0:
            right_ankle = np.array(self.trajectory_data['right_ankle_3d'])
            
            axes[0].plot(timestamps, right_ankle[:, 0], 'r-', label='Right Ankle X', linewidth=2)
            axes[1].plot(timestamps, right_ankle[:, 1], 'r-', label='Right Ankle Y', linewidth=2)
            axes[2].plot(timestamps, right_ankle[:, 2], 'r-', label='Right Ankle Z', linewidth=2)
        
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('X Position (m)')
        axes[0].set_title('Ankle Position X vs Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Y Position (m)')
        axes[1].set_title('Ankle Position Y vs Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Z Position (m)')
        axes[2].set_title('Ankle Position Z vs Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to: {save_path}")
        else:
            plt.show()
    
    def plot_displacement(self, side: str = 'both', save_path: Optional[str] = None):
        """Plot displacement dari initial position"""
        if self.trajectory_data is None:
            print("No trajectory data loaded")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        timestamps = np.array(self.trajectory_data['timestamps'])
        if len(timestamps) > 1:
            timestamps = timestamps - timestamps[0]
        
        if side in ['left', 'both'] and len(self.trajectory_data['left_ankle_3d']) > 0:
            left_ankle = np.array(self.trajectory_data['left_ankle_3d'])
            left_displacement = self.tracker.calculate_displacement(left_ankle)
            
            axes[0].plot(timestamps, left_displacement[:, 0], 'b-', label='Left Ankle ΔX', linewidth=2)
            axes[1].plot(timestamps, left_displacement[:, 1], 'b-', label='Left Ankle ΔY', linewidth=2)
            axes[2].plot(timestamps, left_displacement[:, 2], 'b-', label='Left Ankle ΔZ', linewidth=2)
        
        if side in ['right', 'both'] and len(self.trajectory_data['right_ankle_3d']) > 0:
            right_ankle = np.array(self.trajectory_data['right_ankle_3d'])
            right_displacement = self.tracker.calculate_displacement(right_ankle)
            
            axes[0].plot(timestamps, right_displacement[:, 0], 'r-', label='Right Ankle ΔX', linewidth=2)
            axes[1].plot(timestamps, right_displacement[:, 1], 'r-', label='Right Ankle ΔY', linewidth=2)
            axes[2].plot(timestamps, right_displacement[:, 2], 'r-', label='Right Ankle ΔZ', linewidth=2)
        
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Displacement X (m)')
        axes[0].set_title('Ankle Displacement X vs Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Displacement Y (m)')
        axes[1].set_title('Ankle Displacement Y vs Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Displacement Z (m)')
        axes[2].set_title('Ankle Displacement Z vs Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
    
    def plot_rotation(self, side: str = 'both', save_path: Optional[str] = None):
        """Plot rotation (Euler angles) terhadap waktu"""
        if self.trajectory_data is None:
            print("No trajectory data loaded")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        timestamps = np.array(self.trajectory_data['timestamps'])
        if len(timestamps) > 1:
            timestamps = timestamps - timestamps[0]
        
        if side in ['left', 'both'] and len(self.trajectory_data['left_ankle_rotations']) > 0:
            left_rotations = self.trajectory_data['left_ankle_rotations']
            left_euler = np.array([r['euler_angles'] for r in left_rotations])
            
            axes[0].plot(timestamps[:len(left_euler)], left_euler[:, 0], 'b-', label='Left Ankle Roll (X)', linewidth=2)
            axes[1].plot(timestamps[:len(left_euler)], left_euler[:, 1], 'b-', label='Left Ankle Pitch (Y)', linewidth=2)
            axes[2].plot(timestamps[:len(left_euler)], left_euler[:, 2], 'b-', label='Left Ankle Yaw (Z)', linewidth=2)
        
        if side in ['right', 'both'] and len(self.trajectory_data['right_ankle_rotations']) > 0:
            right_rotations = self.trajectory_data['right_ankle_rotations']
            right_euler = np.array([r['euler_angles'] for r in right_rotations])
            
            axes[0].plot(timestamps[:len(right_euler)], right_euler[:, 0], 'r-', label='Right Ankle Roll (X)', linewidth=2)
            axes[1].plot(timestamps[:len(right_euler)], right_euler[:, 1], 'r-', label='Right Ankle Pitch (Y)', linewidth=2)
            axes[2].plot(timestamps[:len(right_euler)], right_euler[:, 2], 'r-', label='Right Ankle Yaw (Z)', linewidth=2)
        
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Roll (degrees)')
        axes[0].set_title('Ankle Rotation Roll (X-axis) vs Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Pitch (degrees)')
        axes[1].set_title('Ankle Rotation Pitch (Y-axis) vs Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Yaw (degrees)')
        axes[2].set_title('Ankle Rotation Yaw (Z-axis) vs Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
    
    def plot_3d_trajectory(self, side: str = 'both', save_path: Optional[str] = None):
        """Plot 3D trajectory ankle"""
        if self.trajectory_data is None:
            print("No trajectory data loaded")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if side in ['left', 'both'] and len(self.trajectory_data['left_ankle_3d']) > 0:
            left_ankle = np.array(self.trajectory_data['left_ankle_3d'])
            ax.plot(left_ankle[:, 0], left_ankle[:, 1], left_ankle[:, 2], 
                   'b-', label='Left Ankle', linewidth=2, alpha=0.7)
            ax.scatter(left_ankle[0, 0], left_ankle[0, 1], left_ankle[0, 2], 
                      c='green', s=100, marker='o', label='Start')
            ax.scatter(left_ankle[-1, 0], left_ankle[-1, 1], left_ankle[-1, 2], 
                      c='red', s=100, marker='x', label='End')
        
        if side in ['right', 'both'] and len(self.trajectory_data['right_ankle_3d']) > 0:
            right_ankle = np.array(self.trajectory_data['right_ankle_3d'])
            ax.plot(right_ankle[:, 0], right_ankle[:, 1], right_ankle[:, 2], 
                   'r-', label='Right Ankle', linewidth=2, alpha=0.7)
            ax.scatter(right_ankle[0, 0], right_ankle[0, 1], right_ankle[0, 2], 
                      c='green', s=100, marker='o')
            ax.scatter(right_ankle[-1, 0], right_ankle[-1, 1], right_ankle[-1, 2], 
                      c='red', s=100, marker='x')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Ankle 3D Trajectory')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()

