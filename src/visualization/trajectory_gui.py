"""
GUI untuk visualisasi trajectory ankle joint
"""
import sys
import json
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QGroupBox, QGridLayout, 
                             QComboBox, QMessageBox)
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import os
# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tracking.ankle_tracker import AnkleTracker
from visualization.trajectory_visualizer import TrajectoryVisualizer


class TrajectoryPlotWidget(FigureCanvas):
    """Widget untuk plot trajectory"""
    
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
    
    def plot_position_time(self, timestamps, left_ankle, right_ankle, axis: str):
        """Plot position vs time untuk satu axis"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        if left_ankle is not None and len(left_ankle) > 0:
            left_data = left_ankle[:, {'x': 0, 'y': 1, 'z': 2}[axis.lower()]]
            ax.plot(timestamps, left_data, 'b-', label='Left Ankle', linewidth=2)
        
        if right_ankle is not None and len(right_ankle) > 0:
            right_data = right_ankle[:, {'x': 0, 'y': 1, 'z': 2}[axis.lower()]]
            ax.plot(timestamps, right_data, 'r-', label='Right Ankle', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{axis.upper()} Position (m)')
        ax.set_title(f'Ankle Position {axis.upper()} vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_displacement_time(self, timestamps, left_displacement, right_displacement, axis: str):
        """Plot displacement vs time"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        if left_displacement is not None and len(left_displacement) > 0:
            left_data = left_displacement[:, {'x': 0, 'y': 1, 'z': 2}[axis.lower()]]
            ax.plot(timestamps, left_data, 'b-', label='Left Ankle Δ' + axis.upper(), linewidth=2)
        
        if right_displacement is not None and len(right_displacement) > 0:
            right_data = right_displacement[:, {'x': 0, 'y': 1, 'z': 2}[axis.lower()]]
            ax.plot(timestamps, right_data, 'r-', label='Right Ankle Δ' + axis.upper(), linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Displacement {axis.upper()} (m)')
        ax.set_title(f'Ankle Displacement {axis.upper()} vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_rotation_time(self, timestamps, left_euler, right_euler, axis: str):
        """Plot rotation vs time"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        axis_idx = {'roll': 0, 'pitch': 1, 'yaw': 2}[axis.lower()]
        axis_name = {'roll': 'Roll (X)', 'pitch': 'Pitch (Y)', 'yaw': 'Yaw (Z)'}[axis.lower()]
        
        if left_euler is not None and len(left_euler) > 0:
            ax.plot(timestamps[:len(left_euler)], left_euler[:, axis_idx], 
                   'b-', label='Left Ankle ' + axis_name, linewidth=2)
        
        if right_euler is not None and len(right_euler) > 0:
            ax.plot(timestamps[:len(right_euler)], right_euler[:, axis_idx], 
                   'r-', label='Right Ankle ' + axis_name, linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{axis_name} (degrees)')
        ax.set_title(f'Ankle Rotation {axis_name} vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_3d_trajectory(self, left_ankle, right_ankle):
        """Plot 3D trajectory"""
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d')
        
        if left_ankle is not None and len(left_ankle) > 0:
            ax.plot(left_ankle[:, 0], left_ankle[:, 1], left_ankle[:, 2], 
                   'b-', label='Left Ankle', linewidth=2, alpha=0.7)
            ax.scatter(left_ankle[0, 0], left_ankle[0, 1], left_ankle[0, 2], 
                      c='green', s=100, marker='o', label='Start')
            ax.scatter(left_ankle[-1, 0], left_ankle[-1, 1], left_ankle[-1, 2], 
                      c='red', s=100, marker='x', label='End')
        
        if right_ankle is not None and len(right_ankle) > 0:
            ax.plot(right_ankle[:, 0], right_ankle[:, 1], right_ankle[:, 2], 
                   'r-', label='Right Ankle', linewidth=2, alpha=0.7)
            if left_ankle is None or len(left_ankle) == 0:
                ax.scatter(right_ankle[0, 0], right_ankle[0, 1], right_ankle[0, 2], 
                          c='green', s=100, marker='o', label='Start')
                ax.scatter(right_ankle[-1, 0], right_ankle[-1, 1], right_ankle[-1, 2], 
                          c='red', s=100, marker='x', label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Ankle 3D Trajectory')
        ax.legend()
        
        self.fig.tight_layout()
        self.draw()


class TrajectoryWidget(QWidget):
    """Widget untuk analisis dan visualisasi trajectory ankle"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracker = AnkleTracker()
        self.visualizer = TrajectoryVisualizer()
        self.trajectory_data = None
        self.current_data = {}
        
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
        
        # Analysis options
        options_group = QGroupBox("Analysis Options")
        options_layout = QGridLayout()
        
        options_layout.addWidget(QLabel("Side:"), 0, 0)
        self.side_combo = QComboBox()
        self.side_combo.addItems(["Both", "Left", "Right"])
        self.side_combo.currentIndexChanged.connect(self.update_plots)
        options_layout.addWidget(self.side_combo, 0, 1)
        
        options_layout.addWidget(QLabel("Joint:"), 0, 2)
        self.joint_combo = QComboBox()
        self.joint_combo.addItems(["Ankle", "Heel", "Toe", "All"])
        self.joint_combo.currentIndexChanged.connect(self.update_plots)
        options_layout.addWidget(self.joint_combo, 0, 3)
        
        options_layout.addWidget(QLabel("View:"), 1, 0)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Position vs Time", "Displacement vs Time", "Rotation vs Time", "3D Trajectory", "Motion Analysis"])
        self.view_combo.currentIndexChanged.connect(self.update_plots)
        options_layout.addWidget(self.view_combo, 1, 1)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Plot area
        self.plot_widget = TrajectoryPlotWidget(self, width=12, height=8)
        self.plot_widget.setMinimumSize(1000, 600)
        layout.addWidget(self.plot_widget)
        
        # Stats
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel("Load a file to see statistics")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        self.setLayout(layout)
    
    def load_file(self):
        """Load JSON file dan analisis trajectory"""
        filepath, _ = QFileDialog.getOpenFileName(self, "Load JSON File", "", "JSON Files (*.json)")
        if filepath:
            try:
                self.trajectory_data = self.visualizer.load_and_analyze(filepath)
                self.file_label.setText(f"Loaded: {filepath} ({len(self.trajectory_data['timestamps'])} frames)")
                
                # Prepare data
                self.prepare_data()
                
                # Update plots
                self.update_plots()
                
                # Update stats
                self.update_stats()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file:\n{str(e)}")
    
    def prepare_data(self):
        """Prepare data untuk plotting"""
        if self.trajectory_data is None:
            return
        
        timestamps = np.array(self.trajectory_data['timestamps'])
        if len(timestamps) > 1:
            timestamps = timestamps - timestamps[0]
        
        # Ankle
        left_ankle = None
        right_ankle = None
        if len(self.trajectory_data.get('left_ankle_3d', [])) > 0:
            left_ankle = np.array(self.trajectory_data['left_ankle_3d'])
        if len(self.trajectory_data.get('right_ankle_3d', [])) > 0:
            right_ankle = np.array(self.trajectory_data['right_ankle_3d'])
        
        # Heel
        left_heel = None
        right_heel = None
        if len(self.trajectory_data.get('left_heel_3d', [])) > 0:
            left_heel = np.array(self.trajectory_data['left_heel_3d'])
        if len(self.trajectory_data.get('right_heel_3d', [])) > 0:
            right_heel = np.array(self.trajectory_data['right_heel_3d'])
        
        # Toe
        left_toe = None
        right_toe = None
        if len(self.trajectory_data.get('left_toe_3d', [])) > 0:
            left_toe = np.array(self.trajectory_data['left_toe_3d'])
        if len(self.trajectory_data.get('right_toe_3d', [])) > 0:
            right_toe = np.array(self.trajectory_data['right_toe_3d'])
        
        # Calculate displacement untuk semua joints
        left_ankle_displacement = self.tracker.calculate_displacement(left_ankle) if left_ankle is not None else None
        right_ankle_displacement = self.tracker.calculate_displacement(right_ankle) if right_ankle is not None else None
        left_heel_displacement = self.tracker.calculate_displacement(left_heel) if left_heel is not None else None
        right_heel_displacement = self.tracker.calculate_displacement(right_heel) if right_heel is not None else None
        left_toe_displacement = self.tracker.calculate_displacement(left_toe) if left_toe is not None else None
        right_toe_displacement = self.tracker.calculate_displacement(right_toe) if right_toe is not None else None
        
        # Get rotations
        left_euler = None
        right_euler = None
        if len(self.trajectory_data.get('left_ankle_rotations', [])) > 0:
            left_euler = np.array([r['euler_angles'] for r in self.trajectory_data['left_ankle_rotations']])
        if len(self.trajectory_data.get('right_ankle_rotations', [])) > 0:
            right_euler = np.array([r['euler_angles'] for r in self.trajectory_data['right_ankle_rotations']])
        
        # Motion analysis
        motion_analysis = self.tracker.analyze_motion_trials(self.trajectory_data)
        
        self.current_data = {
            'timestamps': timestamps,
            'left_ankle': left_ankle,
            'right_ankle': right_ankle,
            'left_heel': left_heel,
            'right_heel': right_heel,
            'left_toe': left_toe,
            'right_toe': right_toe,
            'left_ankle_displacement': left_ankle_displacement,
            'right_ankle_displacement': right_ankle_displacement,
            'left_heel_displacement': left_heel_displacement,
            'right_heel_displacement': right_heel_displacement,
            'left_toe_displacement': left_toe_displacement,
            'right_toe_displacement': right_toe_displacement,
            'left_euler': left_euler,
            'right_euler': right_euler,
            'motion_analysis': motion_analysis
        }
    
    def update_plots(self):
        """Update plots berdasarkan pilihan"""
        if self.trajectory_data is None:
            return
        
        side = self.side_combo.currentText().lower()
        joint = self.joint_combo.currentText().lower()
        view = self.view_combo.currentText()
        timestamps = self.current_data.get('timestamps')
        
        # Get data based on side and joint
        def get_joint_data(joint_name):
            if joint_name == 'ankle':
                return {
                    'left': self.current_data.get('left_ankle') if side in ['both', 'left'] else None,
                    'right': self.current_data.get('right_ankle') if side in ['both', 'right'] else None,
                    'left_disp': self.current_data.get('left_ankle_displacement') if side in ['both', 'left'] else None,
                    'right_disp': self.current_data.get('right_ankle_displacement') if side in ['both', 'right'] else None
                }
            elif joint_name == 'heel':
                return {
                    'left': self.current_data.get('left_heel') if side in ['both', 'left'] else None,
                    'right': self.current_data.get('right_heel') if side in ['both', 'right'] else None,
                    'left_disp': self.current_data.get('left_heel_displacement') if side in ['both', 'left'] else None,
                    'right_disp': self.current_data.get('right_heel_displacement') if side in ['both', 'right'] else None
                }
            elif joint_name == 'toe':
                return {
                    'left': self.current_data.get('left_toe') if side in ['both', 'left'] else None,
                    'right': self.current_data.get('right_toe') if side in ['both', 'right'] else None,
                    'left_disp': self.current_data.get('left_toe_displacement') if side in ['both', 'left'] else None,
                    'right_disp': self.current_data.get('right_toe_displacement') if side in ['both', 'right'] else None
                }
            else:  # all
                return {
                    'left_ankle': self.current_data.get('left_ankle') if side in ['both', 'left'] else None,
                    'right_ankle': self.current_data.get('right_ankle') if side in ['both', 'right'] else None,
                    'left_heel': self.current_data.get('left_heel') if side in ['both', 'left'] else None,
                    'right_heel': self.current_data.get('right_heel') if side in ['both', 'right'] else None,
                    'left_toe': self.current_data.get('left_toe') if side in ['both', 'left'] else None,
                    'right_toe': self.current_data.get('right_toe') if side in ['both', 'right'] else None
                }
        
        if view == "Position vs Time":
            self.plot_widget.fig.clear()
            axes = self.plot_widget.fig.subplots(3, 1)
            
            if joint == 'all':
                data_dict = get_joint_data('all')
                for i, axis in enumerate(['X', 'Y', 'Z']):
                    for name, data in data_dict.items():
                        if data is not None and len(data) > 0:
                            color = 'b' if 'left' in name else 'r'
                            style = '-' if 'ankle' in name else '--' if 'heel' in name else ':'
                            label = name.replace('_', ' ').title()
                            axes[i].plot(timestamps, data[:, i], color=color, linestyle=style, label=label, linewidth=2)
            else:
                data_dict = get_joint_data(joint)
                left_data = data_dict['left']
                right_data = data_dict['right']
                joint_label = joint.title()
                
                for i, axis in enumerate(['X', 'Y', 'Z']):
                    if left_data is not None and len(left_data) > 0:
                        axes[i].plot(timestamps, left_data[:, i], 'b-', label=f'Left {joint_label}', linewidth=2)
                    if right_data is not None and len(right_data) > 0:
                        axes[i].plot(timestamps, right_data[:, i], 'r-', label=f'Right {joint_label}', linewidth=2)
            
            for i, axis in enumerate(['X', 'Y', 'Z']):
                axes[i].set_xlabel('Time (s)')
                axes[i].set_ylabel(f'{axis} Position (m)')
                axes[i].set_title(f'Position {axis} vs Time')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            self.plot_widget.fig.tight_layout()
            self.plot_widget.draw()
        
        elif view == "Displacement vs Time":
            self.plot_widget.fig.clear()
            axes = self.plot_widget.fig.subplots(3, 1)
            
            if joint == 'all':
                data_dict = get_joint_data('all')
                for i, axis in enumerate(['X', 'Y', 'Z']):
                    for name, data in data_dict.items():
                        if data is not None and len(data) > 0:
                            disp = self.tracker.calculate_displacement(data)
                            color = 'b' if 'left' in name else 'r'
                            style = '-' if 'ankle' in name else '--' if 'heel' in name else ':'
                            label = name.replace('_', ' ').title()
                            axes[i].plot(timestamps, disp[:, i], color=color, linestyle=style, label=label, linewidth=2)
            else:
                data_dict = get_joint_data(joint)
                left_disp = data_dict['left_disp']
                right_disp = data_dict['right_disp']
                joint_label = joint.title()
                
                for i, axis in enumerate(['X', 'Y', 'Z']):
                    if left_disp is not None and len(left_disp) > 0:
                        axes[i].plot(timestamps, left_disp[:, i], 'b-', label=f'Left {joint_label} Δ' + axis, linewidth=2)
                    if right_disp is not None and len(right_disp) > 0:
                        axes[i].plot(timestamps, right_disp[:, i], 'r-', label=f'Right {joint_label} Δ' + axis, linewidth=2)
            
            for i, axis in enumerate(['X', 'Y', 'Z']):
                axes[i].set_xlabel('Time (s)')
                axes[i].set_ylabel(f'Displacement {axis} (m)')
                axes[i].set_title(f'Displacement {axis} vs Time')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            self.plot_widget.fig.tight_layout()
            self.plot_widget.draw()
        
        elif view == "Rotation vs Time":
            # Only for ankle
            self.plot_widget.fig.clear()
            axes = self.plot_widget.fig.subplots(3, 1)
            
            left_euler = self.current_data.get('left_euler') if side in ['both', 'left'] else None
            right_euler = self.current_data.get('right_euler') if side in ['both', 'right'] else None
            
            rotation_names = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
            for i, name in enumerate(rotation_names):
                if left_euler is not None and len(left_euler) > 0:
                    axes[i].plot(timestamps[:len(left_euler)], left_euler[:, i], 
                               'b-', label='Left Ankle ' + name, linewidth=2)
                if right_euler is not None and len(right_euler) > 0:
                    axes[i].plot(timestamps[:len(right_euler)], right_euler[:, i], 
                               'r-', label='Right Ankle ' + name, linewidth=2)
                axes[i].set_xlabel('Time (s)')
                axes[i].set_ylabel(f'{name} (degrees)')
                axes[i].set_title(f'Ankle Rotation {name} vs Time')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            self.plot_widget.fig.tight_layout()
            self.plot_widget.draw()
        
        elif view == "3D Trajectory":
            self.plot_widget.fig.clear()
            ax = self.plot_widget.fig.add_subplot(111, projection='3d')
            
            if joint == 'all':
                data_dict = get_joint_data('all')
                colors = {'ankle': 'blue', 'heel': 'green', 'toe': 'orange'}
                for name, data in data_dict.items():
                    if data is not None and len(data) > 0:
                        color = colors.get(name.split('_')[-1], 'gray')
                        side_label = 'Left' if 'left' in name else 'Right'
                        joint_name = name.split('_')[-1].title()
                        ax.plot(data[:, 0], data[:, 1], data[:, 2], 
                               color=color, label=f'{side_label} {joint_name}', linewidth=2, alpha=0.7)
            else:
                data_dict = get_joint_data(joint)
                left_data = data_dict['left']
                right_data = data_dict['right']
                joint_label = joint.title()
                
                if left_data is not None and len(left_data) > 0:
                    ax.plot(left_data[:, 0], left_data[:, 1], left_data[:, 2], 
                           'b-', label=f'Left {joint_label}', linewidth=2, alpha=0.7)
                    ax.scatter(left_data[0, 0], left_data[0, 1], left_data[0, 2], 
                              c='green', s=100, marker='o', label='Start')
                if right_data is not None and len(right_data) > 0:
                    ax.plot(right_data[:, 0], right_data[:, 1], right_data[:, 2], 
                           'r-', label=f'Right {joint_label}', linewidth=2, alpha=0.7)
                    if left_data is None or len(left_data) == 0:
                        ax.scatter(right_data[0, 0], right_data[0, 1], right_data[0, 2], 
                                  c='green', s=100, marker='o', label='Start')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('3D Trajectory')
            ax.legend()
            self.plot_widget.fig.tight_layout()
            self.plot_widget.draw()
        
        elif view == "Motion Analysis":
            # Plot motion analysis: dorsiflexion/plantarflexion, inversion/eversion
            motion_analysis = self.current_data.get('motion_analysis', {})
            
            self.plot_widget.fig.clear()
            axes = self.plot_widget.fig.subplots(2, 1)
            
            # Dorsiflexion/Plantarflexion
            df_pf = motion_analysis.get('dorsiflexion_plantarflexion', {})
            if 'left' in df_pf and df_pf['left'].get('angles'):
                axes[0].plot(timestamps[:len(df_pf['left']['angles'])], df_pf['left']['angles'], 
                           'b-', label='Left', linewidth=2)
            if 'right' in df_pf and df_pf['right'].get('angles'):
                axes[0].plot(timestamps[:len(df_pf['right']['angles'])], df_pf['right']['angles'], 
                           'r-', label='Right', linewidth=2)
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Angle (degrees)')
            axes[0].set_title('Dorsiflexion/Plantarflexion')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Inversion/Eversion
            inv_ev = motion_analysis.get('inversion_eversion', {})
            if 'left' in inv_ev and inv_ev['left'].get('angles'):
                axes[1].plot(timestamps[:len(inv_ev['left']['angles'])], inv_ev['left']['angles'], 
                           'b-', label='Left', linewidth=2)
            if 'right' in inv_ev and inv_ev['right'].get('angles'):
                axes[1].plot(timestamps[:len(inv_ev['right']['angles'])], inv_ev['right']['angles'], 
                           'r-', label='Right', linewidth=2)
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Angle (degrees)')
            axes[1].set_title('Inversion/Eversion')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            self.plot_widget.fig.tight_layout()
            self.plot_widget.draw()
    
    def update_stats(self):
        """Update statistics display"""
        if self.trajectory_data is None:
            return
        
        stats_text = "Statistics:\n\n"
        
        # Ankle stats
        if len(self.trajectory_data.get('left_ankle_3d', [])) > 0:
            left_ankle = np.array(self.trajectory_data['left_ankle_3d'])
            left_stats = self.tracker.get_trajectory_stats(left_ankle, self.trajectory_data['timestamps'])
            stats_text += "Left Ankle:\n"
            stats_text += f"  Total Distance: {left_stats.get('total_distance', 0):.3f} m\n"
            stats_text += f"  Range X: {left_stats.get('range', [0,0,0])[0]:.3f} m\n"
            stats_text += f"  Range Y: {left_stats.get('range', [0,0,0])[1]:.3f} m\n"
            stats_text += f"  Range Z: {left_stats.get('range', [0,0,0])[2]:.3f} m\n\n"
        
        if len(self.trajectory_data.get('right_ankle_3d', [])) > 0:
            right_ankle = np.array(self.trajectory_data['right_ankle_3d'])
            right_stats = self.tracker.get_trajectory_stats(right_ankle, self.trajectory_data['timestamps'])
            stats_text += "Right Ankle:\n"
            stats_text += f"  Total Distance: {right_stats.get('total_distance', 0):.3f} m\n"
            stats_text += f"  Range X: {right_stats.get('range', [0,0,0])[0]:.3f} m\n"
            stats_text += f"  Range Y: {right_stats.get('range', [0,0,0])[1]:.3f} m\n"
            stats_text += f"  Range Z: {right_stats.get('range', [0,0,0])[2]:.3f} m\n\n"
        
        # Heel stats
        if len(self.trajectory_data.get('left_heel_3d', [])) > 0:
            left_heel = np.array(self.trajectory_data['left_heel_3d'])
            left_heel_stats = self.tracker.get_trajectory_stats(left_heel, self.trajectory_data['timestamps'])
            stats_text += "Left Heel:\n"
            stats_text += f"  Total Distance: {left_heel_stats.get('total_distance', 0):.3f} m\n\n"
        
        if len(self.trajectory_data.get('right_heel_3d', [])) > 0:
            right_heel = np.array(self.trajectory_data['right_heel_3d'])
            right_heel_stats = self.tracker.get_trajectory_stats(right_heel, self.trajectory_data['timestamps'])
            stats_text += "Right Heel:\n"
            stats_text += f"  Total Distance: {right_heel_stats.get('total_distance', 0):.3f} m\n\n"
        
        # Toe stats
        if len(self.trajectory_data.get('left_toe_3d', [])) > 0:
            left_toe = np.array(self.trajectory_data['left_toe_3d'])
            left_toe_stats = self.tracker.get_trajectory_stats(left_toe, self.trajectory_data['timestamps'])
            stats_text += "Left Toe:\n"
            stats_text += f"  Total Distance: {left_toe_stats.get('total_distance', 0):.3f} m\n\n"
        
        if len(self.trajectory_data.get('right_toe_3d', [])) > 0:
            right_toe = np.array(self.trajectory_data['right_toe_3d'])
            right_toe_stats = self.tracker.get_trajectory_stats(right_toe, self.trajectory_data['timestamps'])
            stats_text += "Right Toe:\n"
            stats_text += f"  Total Distance: {right_toe_stats.get('total_distance', 0):.3f} m\n\n"
        
        # Motion analysis stats
        motion_analysis = self.current_data.get('motion_analysis', {})
        if motion_analysis:
            stats_text += "Motion Analysis:\n"
            df_pf = motion_analysis.get('dorsiflexion_plantarflexion', {})
            if 'left' in df_pf:
                stats_text += f"  Left DF/PF Range: {df_pf['left'].get('range', 0):.1f}°\n"
            if 'right' in df_pf:
                stats_text += f"  Right DF/PF Range: {df_pf['right'].get('range', 0):.1f}°\n"
            
            inv_ev = motion_analysis.get('inversion_eversion', {})
            if 'left' in inv_ev:
                stats_text += f"  Left Inv/Ev Range: {inv_ev['left'].get('range', 0):.1f}°\n"
            if 'right' in inv_ev:
                stats_text += f"  Right Inv/Ev Range: {inv_ev['right'].get('range', 0):.1f}°\n"
        
        self.stats_label.setText(stats_text)

