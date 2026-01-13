"""
GUI untuk visualisasi pose data - Widget untuk diintegrasikan
"""
import sys
import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QSlider, QGroupBox, QGridLayout, QComboBox)
from PyQt5.QtCore import Qt, QTimer
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from .visualize_pose import PoseVisualizer


class PosePlotWidget(FigureCanvas):
    """Widget untuk plot pose"""
    
    def __init__(self, parent=None, width=8, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
    
    def plot_2d_pose(self, landmarks_2d, connections, landmark_names):
        """Plot pose 2D"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Plot connections
        for connection in connections:
            if (connection[0] < len(landmarks_2d) and 
                connection[1] < len(landmarks_2d)):
                x_coords = [landmarks_2d[connection[0]][0], landmarks_2d[connection[1]][0]]
                y_coords = [landmarks_2d[connection[0]][1], landmarks_2d[connection[1]][1]]
                ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.6)
        
        # Plot landmarks
        ax.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1], c='r', s=50, zorder=5)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_3d_pose(self, landmarks_3d, connections):
        """Plot pose 3D"""
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d')
        
        # Plot connections
        for connection in connections:
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
        
        # Set equal aspect ratio
        max_range = np.array([
            landmarks_3d[:, 0].max() - landmarks_3d[:, 0].min(),
            landmarks_3d[:, 1].max() - landmarks_3d[:, 1].min(),
            landmarks_3d[:, 2].max() - landmarks_3d[:, 2].min()
        ]).max() / 2.0
        mid_x = (landmarks_3d[:, 0].max() + landmarks_3d[:, 0].min()) * 0.5
        mid_y = (landmarks_3d[:, 1].max() + landmarks_3d[:, 1].min()) * 0.5
        mid_z = (landmarks_3d[:, 2].max() + landmarks_3d[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        self.fig.tight_layout()
        self.draw()


class VisualizeWidget(QWidget):
    """Widget untuk visualisasi - bisa diintegrasikan ke tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualizer = None
        self.current_frame = 0
        self.is_playing = False
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # File selection
        file_group = QGroupBox("File")
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file loaded")
        self.load_btn = QPushButton("Load JSON File")
        self.load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.load_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Plot area
        plot_layout = QHBoxLayout()
        
        self.plot_2d = PosePlotWidget(self, width=6, height=6)
        self.plot_2d.setMinimumSize(600, 600)
        plot_layout.addWidget(self.plot_2d)
        
        self.plot_3d = PosePlotWidget(self, width=6, height=6)
        self.plot_3d.setMinimumSize(600, 600)
        plot_layout.addWidget(self.plot_3d)
        
        layout.addLayout(plot_layout)
        
        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QGridLayout()
        
        # Frame navigation
        controls_layout.addWidget(QLabel("Frame:"), 0, 0)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        controls_layout.addWidget(self.frame_slider, 0, 1)
        
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(0)
        self.frame_spin.valueChanged.connect(self.on_frame_spin_changed)
        controls_layout.addWidget(self.frame_spin, 0, 2)
        
        self.frame_label = QLabel("0 / 0")
        controls_layout.addWidget(self.frame_label, 0, 3)
        
        # Playback controls
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self.prev_frame)
        controls_layout.addWidget(self.prev_btn, 1, 0)
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_btn, 1, 1)
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_frame)
        controls_layout.addWidget(self.next_btn, 1, 2)
        
        # View mode
        controls_layout.addWidget(QLabel("View:"), 2, 0)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["2D", "3D", "Both"])
        self.view_combo.setCurrentIndex(2)
        self.view_combo.currentIndexChanged.connect(self.update_plot)
        controls_layout.addWidget(self.view_combo, 2, 1)
        
        # Body mode (Full Body / Lower Body)
        controls_layout.addWidget(QLabel("Body:"), 3, 0)
        self.body_combo = QComboBox()
        self.body_combo.addItems(["Full Body", "Lower Body"])
        self.body_combo.setCurrentIndex(0)
        self.body_combo.currentIndexChanged.connect(self.update_plot)
        controls_layout.addWidget(self.body_combo, 3, 1)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        self.setLayout(layout)
    
    def load_file(self):
        """Load JSON file"""
        filepath, _ = QFileDialog.getOpenFileName(self, "Load JSON File", "", "JSON Files (*.json)")
        if filepath:
            try:
                self.visualizer = PoseVisualizer(filepath)
                self.file_label.setText(f"Loaded: {filepath} ({len(self.visualizer.data)} frames)")
                
                # Update controls
                max_frames = len(self.visualizer.data) - 1
                self.frame_slider.setMaximum(max_frames)
                self.frame_spin.setMaximum(max_frames)
                self.current_frame = 0
                self.frame_slider.setValue(0)
                self.frame_spin.setValue(0)
                
                self.update_plot()
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Error loading file:\n{str(e)}")
    
    def on_frame_changed(self, value):
        """Frame slider changed"""
        self.current_frame = value
        self.frame_spin.setValue(value)
        self.update_plot()
    
    def on_frame_spin_changed(self, value):
        """Frame spinbox changed"""
        self.current_frame = value
        self.frame_slider.setValue(value)
        self.update_plot()
    
    def prev_frame(self):
        """Previous frame"""
        if self.visualizer and self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.setValue(self.current_frame)
            self.frame_spin.setValue(self.current_frame)
    
    def next_frame(self):
        """Next frame"""
        if self.visualizer and self.current_frame < len(self.visualizer.data) - 1:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
            self.frame_spin.setValue(self.current_frame)
        else:
            self.toggle_play()  # Stop jika sudah di akhir
    
    def toggle_play(self):
        """Toggle play/pause"""
        if not self.visualizer:
            return
        
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.setText("⏸ Pause")
            self.play_timer.start(100)  # 10 FPS
        else:
            self.play_btn.setText("▶ Play")
            self.play_timer.stop()
    
    def filter_landmarks(self, landmarks, body_mode):
        """
        Filter landmarks berdasarkan body mode
        
        Args:
            landmarks: Array of landmarks (N, 2) or (N, 3)
            body_mode: 0 = Full Body, 1 = Lower Body
        
        Returns:
            Filtered landmarks dan mapping indices
        """
        if body_mode == 0:  # Full Body
            return landmarks, list(range(len(landmarks)))
        
        # Lower Body: hanya hip (23, 24), knee (25, 26), ankle (27, 28)
        lower_body_indices = [23, 24, 25, 26, 27, 28]
        
        if len(landmarks) <= max(lower_body_indices):
            return landmarks, list(range(len(landmarks)))
        
        filtered_landmarks = landmarks[lower_body_indices]
        # Create mapping: original index -> new index
        index_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(lower_body_indices)}
        
        return filtered_landmarks, index_mapping
    
    def filter_connections(self, connections, body_mode, index_mapping):
        """
        Filter connections berdasarkan body mode
        
        Args:
            connections: List of (idx1, idx2) tuples
            body_mode: 0 = Full Body, 1 = Lower Body
            index_mapping: Dictionary mapping original indices to new indices
        
        Returns:
            Filtered connections
        """
        if body_mode == 0:  # Full Body
            return connections
        
        # Lower Body connections: hip-hip, hip-knee, knee-ankle
        lower_body_connections = [
            (23, 24),  # hip to hip
            (23, 25),  # left hip to left knee
            (24, 26),  # right hip to right knee
            (25, 27),  # left knee to left ankle
            (26, 28),  # right knee to right ankle
        ]
        
        filtered_connections = []
        for conn in lower_body_connections:
            if conn[0] in index_mapping and conn[1] in index_mapping:
                filtered_connections.append((index_mapping[conn[0]], index_mapping[conn[1]]))
        
        return filtered_connections
    
    def update_plot(self):
        """Update plot dengan frame saat ini"""
        if not self.visualizer or not self.visualizer.data:
            return
        
        if self.current_frame >= len(self.visualizer.data):
            return
        
        frame_data = self.visualizer.data[self.current_frame]
        view_mode = self.view_combo.currentIndex()
        body_mode = self.body_combo.currentIndex()  # 0 = Full Body, 1 = Lower Body
        
        # Update frame label
        self.frame_label.setText(f"{self.current_frame + 1} / {len(self.visualizer.data)}")
        
        # Plot 2D
        if view_mode in [0, 2]:  # 2D or Both
            if 'landmarks_2d' in frame_data and frame_data['landmarks_2d']:
                landmarks_2d = np.array(frame_data['landmarks_2d'])
                
                # Filter landmarks
                filtered_landmarks_2d, index_mapping = self.filter_landmarks(landmarks_2d, body_mode)
                filtered_connections = self.filter_connections(
                    self.visualizer.POSE_CONNECTIONS, body_mode, index_mapping
                )
                
                # Filter landmark names
                if body_mode == 0:
                    filtered_names = self.visualizer.LANDMARK_NAMES
                else:
                    lower_body_names = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 
                                       'left_ankle', 'right_ankle']
                    filtered_names = lower_body_names
                
                self.plot_2d.plot_2d_pose(
                    filtered_landmarks_2d,
                    filtered_connections,
                    filtered_names
                )
                body_title = "Full Body" if body_mode == 0 else "Lower Body"
                self.plot_2d.fig.suptitle(f'Frame {self.current_frame} - 2D Pose ({body_title})', fontsize=12)
            else:
                self.plot_2d.fig.clear()
                self.plot_2d.fig.text(0.5, 0.5, '2D data tidak tersedia', 
                                      ha='center', va='center')
                self.plot_2d.draw()
        
        # Plot 3D
        if view_mode in [1, 2]:  # 3D or Both
            if 'landmarks_3d' in frame_data and frame_data['landmarks_3d']:
                landmarks_3d = np.array(frame_data['landmarks_3d'])
                
                # Normalize dengan hip center dulu (dari original landmarks)
                if len(landmarks_3d) > 23:
                    hip_center = (landmarks_3d[23] + landmarks_3d[24]) / 2.0
                    landmarks_3d = landmarks_3d - hip_center
                else:
                    hip_center = np.mean(landmarks_3d, axis=0)
                    landmarks_3d = landmarks_3d - hip_center
                
                # Apply rotation
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]
                ])
                landmarks_3d = landmarks_3d @ rotation_matrix.T
                
                # Filter landmarks setelah transformasi
                filtered_landmarks_3d, index_mapping = self.filter_landmarks(landmarks_3d, body_mode)
                filtered_connections = self.filter_connections(
                    self.visualizer.POSE_CONNECTIONS, body_mode, index_mapping
                )
                
                self.plot_3d.plot_3d_pose(
                    filtered_landmarks_3d,
                    filtered_connections
                )
                body_title = "Full Body" if body_mode == 0 else "Lower Body"
                self.plot_3d.fig.suptitle(f'Frame {self.current_frame} - 3D Pose ({body_title})', fontsize=12)
            else:
                self.plot_3d.fig.clear()
                self.plot_3d.fig.text(0.5, 0.5, '3D data tidak tersedia', 
                                      ha='center', va='center')
                self.plot_3d.draw()


class VisualizeWindow(QMainWindow):
    """Window untuk visualisasi pose - standalone"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Data Visualizer")
        self.setGeometry(100, 100, 1400, 800)
        
        self.visualize_widget = VisualizeWidget()
        self.setCentralWidget(self.visualize_widget)


def main():
    app = QApplication(sys.argv)
    window = VisualizeWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
