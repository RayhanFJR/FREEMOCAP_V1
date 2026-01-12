"""
Script untuk memvisualisasikan hasil rekaman pose
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import List, Dict, Optional
import sys
import os


class PoseVisualizer:
    """Kelas untuk visualisasi pose data"""
    
    # Landmark names sesuai MediaPipe
    LANDMARK_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear',
        'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
        'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
        'right_index', 'left_thumb', 'right_thumb', 'left_hip',
        'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Connections untuk menggambar skeleton
    POSE_CONNECTIONS = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
        (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
        (9, 10),  # Mouth
        # Torso
        (11, 12),  # Shoulders
        (11, 23), (12, 24),  # Shoulder to hip
        (23, 24),  # Hips
        # Left arm
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        # Right arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        # Left leg
        (23, 25), (25, 27),
        # Right leg
        (24, 26), (26, 28),
    ]
    
    def __init__(self, data_file: str):
        """
        Args:
            data_file: Path ke file JSON hasil recording
        """
        self.data_file = data_file
        self.data = self.load_data()
        
    def load_data(self) -> List[Dict]:
        """Load data dari file JSON"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"File tidak ditemukan: {self.data_file}")
        
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Format data tidak valid. Harus berupa list of frames.")
        
        print(f"Loaded {len(data)} frames dari {self.data_file}")
        return data
    
    def plot_2d_pose(self, frame_idx: int = 0, save_path: Optional[str] = None):
        """Plot pose 2D untuk satu frame dengan koordinat statis"""
        if frame_idx >= len(self.data):
            print(f"Frame {frame_idx} tidak ada. Total frames: {len(self.data)}")
            return
        
        frame_data = self.data[frame_idx]
        
        if 'landmarks_2d' not in frame_data or not frame_data['landmarks_2d']:
            print("Data 2D tidak tersedia untuk frame ini")
            return
        
        landmarks_2d = np.array(frame_data['landmarks_2d'])
        
        # Normalize: set origin ke tengah pose (hip center jika ada)
        if len(landmarks_2d) > 23:
            hip_center = (landmarks_2d[23] + landmarks_2d[24]) / 2.0
            landmarks_2d = landmarks_2d - hip_center
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot connections
        for connection in self.POSE_CONNECTIONS:
            if (connection[0] < len(landmarks_2d) and 
                connection[1] < len(landmarks_2d)):
                x_coords = [landmarks_2d[connection[0]][0], landmarks_2d[connection[1]][0]]
                y_coords = [landmarks_2d[connection[0]][1], landmarks_2d[connection[1]][1]]
                ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.6)
        
        # Plot landmarks
        ax.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1], c='r', s=50, zorder=5)
        
        # Label beberapa landmark penting
        important_landmarks = [0, 11, 12, 23, 24]  # nose, shoulders, hips
        for idx in important_landmarks:
            if idx < len(landmarks_2d):
                ax.annotate(self.LANDMARK_NAMES[idx], 
                          (landmarks_2d[idx][0], landmarks_2d[idx][1]),
                          fontsize=8)
        
        # Set bounds yang konsisten
        max_range = max(
            landmarks_2d[:, 0].max() - landmarks_2d[:, 0].min(),
            landmarks_2d[:, 1].max() - landmarks_2d[:, 1].min()
        ) / 2.0
        if max_range < 50:
            max_range = 200  # Minimum range untuk pixel
        
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(f'Pose 2D - Frame {frame_idx}')
        ax.invert_yaxis()  # Invert karena koordinat image
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Gambar disimpan ke: {save_path}")
        else:
            plt.show()
    
    def plot_3d_pose(self, frame_idx: int = 0, save_path: Optional[str] = None):
        """Plot pose 3D untuk satu frame dengan koordinat statis"""
        if frame_idx >= len(self.data):
            print(f"Frame {frame_idx} tidak ada. Total frames: {len(self.data)}")
            return
        
        frame_data = self.data[frame_idx]
        
        if 'landmarks_3d' not in frame_data or not frame_data['landmarks_3d']:
            print("Data 3D tidak tersedia untuk frame ini")
            return
        
        landmarks_3d = np.array(frame_data['landmarks_3d'])
        
        # Normalize koordinat: set origin ke tengah-tengah pose
        # Gunakan hip center sebagai reference point
        if len(landmarks_3d) > 23:
            # Hip center (antara left_hip dan right_hip)
            hip_center = (landmarks_3d[23] + landmarks_3d[24]) / 2.0
            landmarks_3d = landmarks_3d - hip_center
            
            # Rotasi untuk orientasi yang lebih natural
            # RealSense coordinate: X=right, Y=down, Z=forward
            # Kita rotasi agar Y=up, Z=forward, X=right
            # Rotasi 90 derajat di sumbu X (tukar Y dan Z)
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ])
            landmarks_3d = landmarks_3d @ rotation_matrix.T
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot connections
        for connection in self.POSE_CONNECTIONS:
            if (connection[0] < len(landmarks_3d) and 
                connection[1] < len(landmarks_3d)):
                x_coords = [landmarks_3d[connection[0]][0], landmarks_3d[connection[1]][0]]
                y_coords = [landmarks_3d[connection[0]][1], landmarks_3d[connection[1]][1]]
                z_coords = [landmarks_3d[connection[0]][2], landmarks_3d[connection[1]][2]]
                ax.plot3D(x_coords, y_coords, z_coords, 'b-', linewidth=2, alpha=0.6)
        
        # Plot landmarks
        ax.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2], 
                  c='r', s=50, zorder=5)
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(f'Pose 3D - Frame {frame_idx}')
        
        # Set equal aspect ratio dengan bounds yang konsisten
        max_range = np.array([
            landmarks_3d[:, 0].max() - landmarks_3d[:, 0].min(),
            landmarks_3d[:, 1].max() - landmarks_3d[:, 1].min(),
            landmarks_3d[:, 2].max() - landmarks_3d[:, 2].min()
        ]).max() / 2.0
        
        # Pastikan ada range minimum
        if max_range < 0.1:
            max_range = 0.5
        
        # Set limits dengan origin di tengah
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        # Set view angle yang lebih natural (elevation, azimuth)
        # elev: tinggi pandangan (0=horizontal, 90=atas)
        # azim: rotasi horizontal (0=depan, 90=kanan)
        ax.view_init(elev=10, azim=45)  # Sudut yang lebih natural
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Gambar disimpan ke: {save_path}")
        else:
            plt.show()
    
    def animate_2d(self, save_path: Optional[str] = None, interval: int = 100):
        """Animasi pose 2D untuk semua frames dengan koordinat statis"""
        if not self.data:
            print("Tidak ada data untuk dianimasikan")
            return
        
        # Normalize semua frames: set origin ke hip center
        normalized_data = []
        for frame_data in self.data:
            if 'landmarks_2d' in frame_data and frame_data['landmarks_2d']:
                landmarks_2d = np.array(frame_data['landmarks_2d'])
                # Normalize dengan hip center
                if len(landmarks_2d) > 23:
                    hip_center = (landmarks_2d[23] + landmarks_2d[24]) / 2.0
                    landmarks_2d = landmarks_2d - hip_center
                normalized_data.append(landmarks_2d)
            else:
                normalized_data.append(None)
        
        # Get bounds dari semua normalized frames
        all_x = []
        all_y = []
        for landmarks_2d in normalized_data:
            if landmarks_2d is not None:
                all_x.extend(landmarks_2d[:, 0])
                all_y.extend(landmarks_2d[:, 1])
        
        if not all_x:
            print("Tidak ada data 2D untuk dianimasikan")
            return
        
        # Hitung max range untuk bounds yang konsisten
        max_range = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y)
        ) / 2.0
        
        if max_range < 50:
            max_range = 200  # Minimum range
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set bounds yang konsisten (statis)
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.invert_yaxis()
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('Pose 2D Animation')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        def animate(frame_idx):
            ax.clear()
            
            # Restore bounds
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.invert_yaxis()
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_title(f'Pose 2D Animation - Frame {frame_idx}/{len(self.data)}')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            landmarks_2d = normalized_data[frame_idx]
            if landmarks_2d is None:
                return
            
            # Plot connections
            for connection in self.POSE_CONNECTIONS:
                if (connection[0] < len(landmarks_2d) and 
                    connection[1] < len(landmarks_2d)):
                    x_coords = [landmarks_2d[connection[0]][0], landmarks_2d[connection[1]][0]]
                    y_coords = [landmarks_2d[connection[0]][1], landmarks_2d[connection[1]][1]]
                    ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.6)
            
            # Plot landmarks
            ax.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1], c='r', s=50, zorder=5)
        
        anim = animation.FuncAnimation(fig, animate, frames=len(self.data), 
                                      interval=interval, repeat=True)
        
        if save_path:
            print(f"Menyimpan animasi ke: {save_path}")
            anim.save(save_path, writer='pillow', fps=1000//interval)
        else:
            plt.show()
    
    def animate_3d(self, save_path: Optional[str] = None, interval: int = 100):
        """Animasi pose 3D untuk semua frames dengan koordinat statis dan interaktif"""
        if not self.data:
            print("Tidak ada data untuk dianimasikan")
            return
        
        # Check apakah ada data 3D
        has_3d = any('landmarks_3d' in frame_data and frame_data['landmarks_3d'] 
                    for frame_data in self.data)
        
        if not has_3d:
            print("Tidak ada data 3D untuk dianimasikan")
            return
        
        # Normalize semua frames: set origin ke hip center untuk setiap frame
        normalized_data = []
        # Rotasi untuk orientasi yang lebih natural (RealSense coordinate system)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        
        for frame_data in self.data:
            if 'landmarks_3d' in frame_data and frame_data['landmarks_3d']:
                landmarks_3d = np.array(frame_data['landmarks_3d'])
                # Normalize dengan hip center
                if len(landmarks_3d) > 23:
                    hip_center = (landmarks_3d[23] + landmarks_3d[24]) / 2.0
                    landmarks_3d = landmarks_3d - hip_center
                    # Apply rotation untuk orientasi yang lebih natural
                    landmarks_3d = landmarks_3d @ rotation_matrix.T
                normalized_data.append(landmarks_3d)
            else:
                normalized_data.append(None)
        
        # Get bounds dari semua normalized data
        all_x, all_y, all_z = [], [], []
        for landmarks_3d in normalized_data:
            if landmarks_3d is not None:
                all_x.extend(landmarks_3d[:, 0])
                all_y.extend(landmarks_3d[:, 1])
                all_z.extend(landmarks_3d[:, 2])
        
        if not all_x:
            print("Tidak ada data valid untuk animasi")
            return
        
        # Hitung max range untuk bounds yang konsisten
        max_range = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            max(all_z) - min(all_z)
        ) / 2.0
        
        # Pastikan ada range minimum
        if max_range < 0.1:
            max_range = 0.5
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set initial view angle
        ax.view_init(elev=10, azim=45)
        
        # Set bounds yang konsisten (statis)
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        # Store current view angles - menggunakan list untuk mutable reference
        view_elev = [10]
        view_azim = [45]
        
        def animate(frame_idx):
            # Simpan view angle SAAT INI sebelum clear (dari frame sebelumnya)
            try:
                view_elev[0] = ax.elev
                view_azim[0] = ax.azim
            except:
                pass
            
            # Clear axes
            ax.clear()
            
            landmarks_3d = normalized_data[frame_idx]
            if landmarks_3d is None:
                return
            
            # Plot connections
            for connection in self.POSE_CONNECTIONS:
                if (connection[0] < len(landmarks_3d) and 
                    connection[1] < len(landmarks_3d)):
                    x_coords = [landmarks_3d[connection[0]][0], landmarks_3d[connection[1]][0]]
                    y_coords = [landmarks_3d[connection[0]][1], landmarks_3d[connection[1]][1]]
                    z_coords = [landmarks_3d[connection[0]][2], landmarks_3d[connection[1]][2]]
                    ax.plot3D(x_coords, y_coords, z_coords, 'b-', linewidth=2, alpha=0.6)
            
            # Plot landmarks
            ax.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2], 
                      c='r', s=50, zorder=5)
            
            # Restore bounds (statis)
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_zlabel('Z (meters)')
            ax.set_title(f'Pose 3D Animation - Frame {frame_idx}/{len(self.data)} (Drag untuk rotate)')
            
            # Restore view angle - gunakan yang tersimpan
            ax.view_init(elev=view_elev[0], azim=view_azim[0])
        
        # Event handler untuk rotasi - update view angle saat user rotate
        # Catatan: view angle akan diupdate oleh event handler saat user drag,
        # dan akan diambil oleh animate() sebelum clear pada frame berikutnya
        
        anim = animation.FuncAnimation(fig, animate, frames=len(self.data), 
                                      interval=interval, repeat=True, blit=False)
        
        # Enable interactive mode
        plt.ion()
        
        if save_path:
            print(f"Menyimpan animasi ke: {save_path}")
            # Untuk save, gunakan view angle tetap
            ax.view_init(elev=10, azim=45)
            anim.save(save_path, writer='pillow', fps=1000//interval)
        else:
            print("Tips: Drag mouse untuk rotate view saat animasi berjalan")
            plt.show(block=True)
    
    def plot_trajectory(self, landmark_idx: int = 0, save_path: Optional[str] = None):
        """Plot trajectory dari satu landmark sepanjang waktu"""
        if landmark_idx >= len(self.LANDMARK_NAMES):
            print(f"Landmark index tidak valid. Max: {len(self.LANDMARK_NAMES)-1}")
            return
        
        # Extract trajectory
        trajectory_2d = []
        trajectory_3d = []
        timestamps = []
        
        for frame_data in self.data:
            if 'timestamp' in frame_data:
                timestamps.append(frame_data['timestamp'])
            else:
                timestamps.append(len(timestamps))
            
            if 'landmarks_2d' in frame_data and frame_data['landmarks_2d']:
                landmarks_2d = np.array(frame_data['landmarks_2d'])
                if landmark_idx < len(landmarks_2d):
                    trajectory_2d.append(landmarks_2d[landmark_idx])
                else:
                    trajectory_2d.append([0, 0])
            else:
                trajectory_2d.append([0, 0])
            
            if 'landmarks_3d' in frame_data and frame_data['landmarks_3d']:
                landmarks_3d = np.array(frame_data['landmarks_3d'])
                if landmark_idx < len(landmarks_3d):
                    trajectory_3d.append(landmarks_3d[landmark_idx])
                else:
                    trajectory_3d.append([0, 0, 0])
            else:
                trajectory_3d.append([0, 0, 0])
        
        trajectory_2d = np.array(trajectory_2d)
        trajectory_3d = np.array(trajectory_3d)
        
        # Normalize timestamps
        if timestamps:
            timestamps = np.array(timestamps)
            timestamps = timestamps - timestamps[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 2D trajectory
        axes[0, 0].plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'b-', alpha=0.5)
        axes[0, 0].scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], c='g', s=100, label='Start', zorder=5)
        axes[0, 0].scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], c='r', s=100, label='End', zorder=5)
        axes[0, 0].set_xlabel('X (pixels)')
        axes[0, 0].set_ylabel('Y (pixels)')
        axes[0, 0].set_title(f'2D Trajectory - {self.LANDMARK_NAMES[landmark_idx]}')
        axes[0, 0].invert_yaxis()
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot X vs time
        axes[0, 1].plot(timestamps, trajectory_2d[:, 0], 'r-', label='X')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('X (pixels)')
        axes[0, 1].set_title('X Position vs Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot Y vs time
        axes[1, 0].plot(timestamps, trajectory_2d[:, 1], 'g-', label='Y')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Y (pixels)')
        axes[1, 0].set_title('Y Position vs Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 3D trajectory jika tersedia
        if trajectory_3d.shape[1] == 3 and np.any(trajectory_3d):
            ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
            ax_3d.plot3D(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], 'b-', alpha=0.5)
            ax_3d.scatter(trajectory_3d[0, 0], trajectory_3d[0, 1], trajectory_3d[0, 2], 
                         c='g', s=100, label='Start', zorder=5)
            ax_3d.scatter(trajectory_3d[-1, 0], trajectory_3d[-1, 1], trajectory_3d[-1, 2], 
                         c='r', s=100, label='End', zorder=5)
            ax_3d.set_xlabel('X (meters)')
            ax_3d.set_ylabel('Y (meters)')
            ax_3d.set_zlabel('Z (meters)')
            ax_3d.set_title('3D Trajectory')
            ax_3d.legend()
        else:
            axes[1, 1].text(0.5, 0.5, '3D data tidak tersedia', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Gambar disimpan ke: {save_path}")
        else:
            plt.show()


def main():
    """Main function untuk command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python visualize_pose.py <json_file> [options]")
        print("\nOptions:")
        print("  --frame <n>        : Plot frame ke-n (2D)")
        print("  --frame3d <n>      : Plot frame ke-n (3D)")
        print("  --animate2d        : Animate semua frames (2D)")
        print("  --animate3d        : Animate semua frames (3D)")
        print("  --trajectory <n>   : Plot trajectory landmark ke-n")
        print("  --save <path>      : Save gambar/animasi ke file")
        return
    
    json_file = sys.argv[1]
    
    try:
        visualizer = PoseVisualizer(json_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Parse arguments
    save_path = None
    if '--save' in sys.argv:
        idx = sys.argv.index('--save')
        if idx + 1 < len(sys.argv):
            save_path = sys.argv[idx + 1]
    
    if '--frame' in sys.argv:
        idx = sys.argv.index('--frame')
        frame_idx = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 0
        visualizer.plot_2d_pose(frame_idx, save_path)
    elif '--frame3d' in sys.argv:
        idx = sys.argv.index('--frame3d')
        frame_idx = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 0
        visualizer.plot_3d_pose(frame_idx, save_path)
    elif '--animate2d' in sys.argv:
        visualizer.animate_2d(save_path)
    elif '--animate3d' in sys.argv:
        visualizer.animate_3d(save_path)
    elif '--trajectory' in sys.argv:
        idx = sys.argv.index('--trajectory')
        landmark_idx = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 0
        visualizer.plot_trajectory(landmark_idx, save_path)
    else:
        # Default: plot frame pertama 2D
        visualizer.plot_2d_pose(0, save_path)


if __name__ == '__main__':
    main()

