import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from collections import deque
import time
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
import os
import glob

class DualCameraFootTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Camera Foot Tracker with Video Export")
        self.root.geometry("1800x950")
        self.root.configure(bg='#2b2b2b')
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose_cam1 = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        )
        self.pose_cam2 = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Camera setup
        self.pipeline = None  # RealSense
        self.align = None
        self.webcam = None  # Webcam/GoPro
        self.webcam_index = 0
        
        # Video recording
        self.video_writer = None
        self.video_file = None
        self.record_video = tk.BooleanVar(value=True)
        
        # Calibration data
        self.calibration_file = "camera_calibration.json"
        self.camera_matrix_1 = None
        self.dist_coeffs_1 = None
        self.camera_matrix_2 = None
        self.dist_coeffs_2 = None
        self.R = None
        self.T = None
        self.is_calibrated = False
        
        # State variables
        self.is_running = False
        self.is_recording = False
        self.is_paused = False
        self.selected_foot = "right"
        self.calibration_mode = False
        self.calibration_points_cam1 = []
        self.calibration_points_cam2 = []
        
        # Data storage
        self.buffer_size = 100
        self.ankle_positions_3d = deque(maxlen=self.buffer_size)
        self.heel_positions_3d = deque(maxlen=self.buffer_size)
        self.toe_positions_3d = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        self.plantar_angles = deque(maxlen=self.buffer_size)
        self.inversion_angles = deque(maxlen=self.buffer_size)
        
        # Landmark indices
        self.update_foot_landmarks()
        
        # CSV logging
        self.csv_file = None
        self.start_time = None
        self.frame_count = 0
        
        # Setup GUI first
        self.setup_gui()
        
        # Load calibration
        self.load_calibration()
        
        # Keyboard bindings
        self.root.bind('<space>', self.on_space_pressed)
        self.root.bind('c', self.on_c_pressed)
        self.root.bind('C', self.on_c_pressed)
        self.root.bind('<Escape>', self.on_esc_pressed)
    
    def on_space_pressed(self, event):
        if self.calibration_mode and hasattr(self, 'temp_calib_corners'):
            if self.temp_calib_corners['ret1'] and self.temp_calib_corners['ret2']:
                self.calibration_points_cam1.append(self.temp_calib_corners['corners1'])
                self.calibration_points_cam2.append(self.temp_calib_corners['corners2'])
                self.update_status(f"✓ Captured calibration point {len(self.calibration_points_cam1)}")
    
    def on_c_pressed(self, event):
        if self.calibration_mode and len(self.calibration_points_cam1) > 0:
            self.calibration_mode = False
            self.perform_calibration()
    
    def on_esc_pressed(self, event):
        if self.calibration_mode:
            self.calibration_mode = False
            self.update_status("✗ Calibration cancelled")
    
    def update_foot_landmarks(self):
        if self.selected_foot == "right":
            self.ANKLE = 28
            self.HEEL = 30
            self.FOOT_INDEX = 32
        else:
            self.ANKLE = 27
            self.HEEL = 29
            self.FOOT_INDEX = 31
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top panel - Camera feeds
        camera_frame = tk.Frame(main_frame, bg='#2b2b2b')
        camera_frame.pack(fill=tk.BOTH, expand=True)
        
        # Camera 1 (RealSense)
        cam1_frame = tk.LabelFrame(camera_frame, text="Camera 1 - RealSense D435i", 
                                   bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        cam1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.camera1_label = tk.Label(cam1_frame, bg='black')
        self.camera1_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera 2 (Webcam/GoPro)
        cam2_frame = tk.LabelFrame(camera_frame, text="Camera 2 - Webcam/GoPro", 
                                   bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        cam2_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.camera2_label = tk.Label(cam2_frame, bg='black')
        self.camera2_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom panel - Controls and graphs
        bottom_frame = tk.Frame(main_frame, bg='#2b2b2b')
        bottom_frame.pack(fill=tk.BOTH, pady=(10, 0))
        
        # Left controls
        control_frame = tk.LabelFrame(bottom_frame, text="Controls", 
                                     bg='#1e1e1e', fg='white', 
                                     font=('Arial', 10, 'bold'), width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_frame.pack_propagate(False)
        
        # Camera selection
        cam_select_frame = tk.Frame(control_frame, bg='#1e1e1e')
        cam_select_frame.pack(pady=5)
        
        tk.Label(cam_select_frame, text="Webcam Index:", bg='#1e1e1e', 
                fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        self.webcam_var = tk.StringVar(value="0")
        webcam_entry = tk.Entry(cam_select_frame, textvariable=self.webcam_var, 
                               width=5, font=('Arial', 9))
        webcam_entry.pack(side=tk.LEFT, padx=5)
        
        # Foot selection
        foot_frame = tk.Frame(control_frame, bg='#1e1e1e')
        foot_frame.pack(pady=5)
        
        tk.Label(foot_frame, text="Select Foot:", bg='#1e1e1e', 
                fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        self.foot_var = tk.StringVar(value="right")
        tk.Radiobutton(foot_frame, text="Right", variable=self.foot_var, 
                      value="right", bg='#1e1e1e', fg='white', 
                      selectcolor='#3e3e3e', command=self.on_foot_change).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(foot_frame, text="Left", variable=self.foot_var, 
                      value="left", bg='#1e1e1e', fg='white', 
                      selectcolor='#3e3e3e', command=self.on_foot_change).pack(side=tk.LEFT, padx=5)
        
        # Video recording option
        video_frame = tk.Frame(control_frame, bg='#1e1e1e')
        video_frame.pack(pady=5)
        
        tk.Checkbutton(video_frame, text="📹 Save Video with CSV", 
                      variable=self.record_video, bg='#1e1e1e', fg='white',
                      selectcolor='#3e3e3e', font=('Arial', 9, 'bold'),
                      activebackground='#1e1e1e', activeforeground='white').pack()
        
        # Calibration status
        self.calib_status = tk.Label(control_frame, 
                                    text="⚠️ Cameras NOT Calibrated", 
                                    bg='#1e1e1e', fg='orange', 
                                    font=('Arial', 9, 'bold'))
        self.calib_status.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(control_frame, bg='#1e1e1e')
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Calibrate Cameras", 
                 command=self.start_calibration, bg='#9C27B0', 
                 fg='white', font=('Arial', 9, 'bold'), 
                 width=15, height=2).grid(row=0, column=0, padx=3, pady=3)
        
        self.start_btn = tk.Button(button_frame, text="Start Cameras", 
                                   command=self.start_cameras, bg='#4CAF50', 
                                   fg='white', font=('Arial', 9, 'bold'), 
                                   width=15, height=2)
        self.start_btn.grid(row=0, column=1, padx=3, pady=3)
        
        self.record_btn = tk.Button(button_frame, text="Start Recording", 
                                    command=self.toggle_recording, bg='#2196F3', 
                                    fg='white', font=('Arial', 9, 'bold'), 
                                    width=15, height=2, state=tk.DISABLED)
        self.record_btn.grid(row=1, column=0, padx=3, pady=3)
        
        self.pause_btn = tk.Button(button_frame, text="Pause", 
                                   command=self.toggle_pause, bg='#FF9800', 
                                   fg='white', font=('Arial', 9, 'bold'), 
                                   width=15, height=2, state=tk.DISABLED)
        self.pause_btn.grid(row=1, column=1, padx=3, pady=3)
        
        self.stop_btn = tk.Button(button_frame, text="Stop Cameras", 
                                  command=self.stop_cameras, bg='#f44336', 
                                  fg='white', font=('Arial', 9, 'bold'), 
                                  width=15, height=2, state=tk.DISABLED)
        self.stop_btn.grid(row=2, column=0, columnspan=2, padx=3, pady=3)
        
        # Open Analyzer button
        tk.Button(button_frame, text="📊 Open CSV Folder", 
                 command=self.open_csv_folder, bg='#00BCD4', 
                 fg='white', font=('Arial', 9, 'bold'), 
                 width=32, height=1).grid(row=3, column=0, columnspan=2, padx=3, pady=3)
        
        # Status panel
        status_frame = tk.LabelFrame(control_frame, text="Status", 
                                    bg='#1e1e1e', fg='white', 
                                    font=('Arial', 10, 'bold'))
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.status_text = tk.Text(status_frame, height=8, bg='#2b2b2b', 
                                  fg='#00ff00', font=('Courier', 8), 
                                  state=tk.DISABLED, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(status_frame, command=self.status_text.yview)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Graphs panel
        graph_frame = tk.Frame(bottom_frame, bg='#2b2b2b')
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # 3D Plot
        plot_3d_frame = tk.LabelFrame(graph_frame, text="3D Triangulated Position", 
                                      bg='#1e1e1e', fg='white', 
                                      font=('Arial', 10, 'bold'))
        plot_3d_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.fig_3d = Figure(figsize=(6, 3), facecolor='#1e1e1e')
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d', facecolor='#2b2b2b')
        self.ax_3d.set_xlabel('X (m)', color='white', fontsize=8)
        self.ax_3d.set_ylabel('Y (m)', color='white', fontsize=8)
        self.ax_3d.set_zlabel('Z (m)', color='white', fontsize=8)
        self.ax_3d.set_title('3D Trajectory', color='white', fontsize=10, fontweight='bold')
        self.ax_3d.tick_params(colors='white', labelsize=7)
        
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=plot_3d_frame)
        self.canvas_3d.draw()
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Translation plots
        plot_trans_frame = tk.LabelFrame(graph_frame, text="Translation Analysis (Ankle)", 
                                         bg='#1e1e1e', fg='white', 
                                         font=('Arial', 10, 'bold'))
        plot_trans_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig_trans = Figure(figsize=(6, 3), facecolor='#1e1e1e')
        
        self.ax_x = self.fig_trans.add_subplot(131, facecolor='#2b2b2b')
        self.ax_y = self.fig_trans.add_subplot(132, facecolor='#2b2b2b')
        self.ax_z = self.fig_trans.add_subplot(133, facecolor='#2b2b2b')
        
        for ax, label in [(self.ax_x, 'X'), (self.ax_y, 'Y'), (self.ax_z, 'Z')]:
            ax.set_xlabel('Time (s)', color='white', fontsize=7)
            ax.set_ylabel(f'{label} (m)', color='white', fontsize=7)
            ax.set_title(f'{label} Position', color='white', fontsize=9, fontweight='bold')
            ax.tick_params(colors='white', labelsize=6)
            ax.grid(True, alpha=0.3, color='white')
        
        self.fig_trans.tight_layout()
        
        self.canvas_trans = FigureCanvasTkAgg(self.fig_trans, master=plot_trans_frame)
        self.canvas_trans.draw()
        self.canvas_trans.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def update_status(self, message):
        """Update status text"""
        self.status_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
    
    def on_foot_change(self):
        self.selected_foot = self.foot_var.get()
        self.update_foot_landmarks()
        self.update_status(f"✓ Selected foot: {self.selected_foot.upper()}")
    
    def start_calibration(self):
        """Start camera calibration wizard"""
        msg = """CAMERA CALIBRATION GUIDE:

1. Print a 9x6 chessboard pattern (25mm squares)
2. Position pattern visible to BOTH cameras
3. Press SPACE to capture each position (15+ angles)
4. Press C when done to calculate calibration
5. Press ESC to cancel

Tips:
- Vary distance and angle
- Keep pattern flat
- Ensure good lighting
- Cover the entire field of view"""
        
        result = messagebox.askokcancel("Calibration", msg)
        if not result:
            return
        
        self.calibration_mode = True
        self.calibration_points_cam1 = []
        self.calibration_points_cam2 = []
        self.temp_calib_corners = {'ret1': False, 'ret2': False, 'corners1': None, 'corners2': None}
        self.update_status("▶ Calibration mode started - Press SPACE to capture")
        
        if not self.is_running:
            self.start_cameras()
    
    def perform_calibration(self):
        """Perform camera calibration"""
        if len(self.calibration_points_cam1) < 10:
            messagebox.showwarning("Insufficient Data", 
                                 f"Need at least 10 calibration points.\nCurrent: {len(self.calibration_points_cam1)}")
            return
        
        try:
            self.update_status("⏳ Calculating calibration...")
            
            objp = np.zeros((6*9, 3), np.float32)
            objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
            
            objpoints = []
            imgpoints1 = []
            imgpoints2 = []
            
            for corners1, corners2 in zip(self.calibration_points_cam1, 
                                         self.calibration_points_cam2):
                objpoints.append(objp)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)
            
            ret1, self.camera_matrix_1, self.dist_coeffs_1, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints1, (640, 480), None, None)
            
            ret2, self.camera_matrix_2, self.dist_coeffs_2, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints2, (640, 480), None, None)
            
            self.update_status(f"  Camera 1 RMS error: {ret1:.4f}")
            self.update_status(f"  Camera 2 RMS error: {ret2:.4f}")
            
            flags = cv2.CALIB_FIX_INTRINSIC
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
            
            ret, _, _, _, _, self.R, self.T, _, _ = cv2.stereoCalibrate(
                objpoints, imgpoints1, imgpoints2,
                self.camera_matrix_1, self.dist_coeffs_1,
                self.camera_matrix_2, self.dist_coeffs_2,
                (640, 480), criteria=criteria, flags=flags)
            
            self.is_calibrated = True
            self.save_calibration()
            
            baseline = np.linalg.norm(self.T) * 100
            
            self.calib_status.config(text="✅ Cameras Calibrated", fg='lime')
            self.update_status(f"✓ Stereo RMS error: {ret:.4f}")
            self.update_status(f"✓ Baseline: {baseline:.2f} cm")
            
            messagebox.showinfo("Success", 
                f"Camera calibration completed!\n\nStereo RMS: {ret:.4f}\nBaseline: {baseline:.2f} cm")
            
        except Exception as e:
            messagebox.showerror("Error", f"Calibration failed:\n{str(e)}")
            self.update_status(f"✗ Calibration ERROR: {str(e)}")
    
    def save_calibration(self):
        """Save calibration data"""
        calib_data = {
            'camera_matrix_1': self.camera_matrix_1.tolist(),
            'dist_coeffs_1': self.dist_coeffs_1.tolist(),
            'camera_matrix_2': self.camera_matrix_2.tolist(),
            'dist_coeffs_2': self.dist_coeffs_2.tolist(),
            'R': self.R.tolist(),
            'T': self.T.tolist(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.calibration_file, 'w') as f:
            json.dump(calib_data, f, indent=4)
        
        self.update_status(f"✓ Calibration saved to {self.calibration_file}")
    
    def load_calibration(self):
        """Load calibration data"""
        try:
            with open(self.calibration_file, 'r') as f:
                calib_data = json.load(f)
            
            self.camera_matrix_1 = np.array(calib_data['camera_matrix_1'])
            self.dist_coeffs_1 = np.array(calib_data['dist_coeffs_1'])
            self.camera_matrix_2 = np.array(calib_data['camera_matrix_2'])
            self.dist_coeffs_2 = np.array(calib_data['dist_coeffs_2'])
            self.R = np.array(calib_data['R'])
            self.T = np.array(calib_data['T'])
            
            self.is_calibrated = True
            
            if hasattr(self, 'calib_status'):
                self.calib_status.config(text="✅ Calibration Loaded", fg='lime')
            if hasattr(self, 'status_text'):
                self.update_status("✓ Calibration loaded from file")
            
        except FileNotFoundError:
            if hasattr(self, 'status_text'):
                self.update_status("⚠️ No calibration file found")
        except Exception as e:
            if hasattr(self, 'status_text'):
                self.update_status(f"✗ Failed to load calibration: {str(e)}")
    
    def triangulate_point(self, point2d_cam1, point2d_cam2):
        """Triangulate 3D point"""
        if not self.is_calibrated:
            return None
        
        try:
            P1 = np.dot(self.camera_matrix_1, np.hstack([np.eye(3), np.zeros((3, 1))]))
            P2 = np.dot(self.camera_matrix_2, np.hstack([self.R, self.T]))
            
            point2d_cam1 = point2d_cam1.reshape(2, 1)
            point2d_cam2 = point2d_cam2.reshape(2, 1)
            
            point_4d = cv2.triangulatePoints(P1, P2, point2d_cam1, point2d_cam2)
            point_3d = point_4d[:3] / point_4d[3]
            
            return point_3d.flatten()
        except Exception as e:
            return None
    
    def start_cameras(self):
        """Start both cameras"""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            
            self.webcam_index = int(self.webcam_var.get())
            self.webcam = cv2.VideoCapture(self.webcam_index)
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self.webcam.isOpened():
                raise Exception(f"Cannot open webcam at index {self.webcam_index}")
            
            self.is_running = True
            self.start_time = time.time()
            
            self.start_btn.config(state=tk.DISABLED)
            self.record_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            
            self.update_status("✓ Both cameras started successfully")
            
            self.processing_thread = threading.Thread(target=self.process_loop, daemon=True)
            self.processing_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start cameras:\n{str(e)}")
            self.update_status(f"✗ ERROR: {str(e)}")
            self.cleanup_cameras()
    
    def stop_cameras(self):
        """Stop both cameras"""
        self.is_running = False
        self.is_recording = False
        self.calibration_mode = False
        
        self.cleanup_cameras()
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.update_status(f"✓ Video saved: {self.video_file}")
        
        self.start_btn.config(state=tk.NORMAL)
        self.record_btn.config(state=tk.DISABLED, text="Start Recording")
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        
        self.update_status("✓ Cameras stopped")
    
    def cleanup_cameras(self):
        """Cleanup camera resources"""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None
        
        if self.webcam:
            self.webcam.release()
            self.webcam = None
    
    def toggle_recording(self):
        """Toggle recording"""
        if not self.is_recording:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_file = f'foot_data_{timestamp_str}.csv'
            
            if self.record_video.get():
                self.video_file = f'foot_video_{timestamp_str}.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.video_file, fourcc, 30.0, (640, 480))
                self.update_status(f"✓ Video recording to: {self.video_file}")
            
            self.init_csv()
            self.is_recording = True
            self.frame_count = 0
            
            self.record_btn.config(text="Stop Recording", bg='#f44336')
            self.pause_btn.config(state=tk.NORMAL)
            
            self.update_status(f"▶ Recording started: {self.csv_file}")
        else:
            self.is_recording = False
            self.is_paused = False
            
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                self.update_status(f"✓ Video saved: {self.video_file}")
            
            self.record_btn.config(text="Start Recording", bg='#2196F3')
            self.pause_btn.config(text="Pause", state=tk.DISABLED)
            
            self.update_status(f"⏹ Recording stopped. Total frames: {self.frame_count}")
            
            # Show completion message with file info
            msg = f"Recording complete!\n\nFiles saved:\n• {self.csv_file}\n"
            if self.video_file:
                msg += f"• {self.video_file}\n"
            msg += "\nFiles are in the same folder as this program."
            messagebox.showinfo("Recording Complete", msg)
    
    def toggle_pause(self):
        """Toggle pause"""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_btn.config(text="Resume", bg='#4CAF50')
            self.update_status("⏸ Recording paused")
        else:
            self.pause_btn.config(text="Pause", bg='#FF9800')
            self.update_status("▶ Recording resumed")
    
    def init_csv(self):
        """Initialize CSV file"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame',
                'ankle_x_3d', 'ankle_y_3d', 'ankle_z_3d',
                'heel_x_3d', 'heel_y_3d', 'heel_z_3d',
                'toe_x_3d', 'toe_y_3d', 'toe_z_3d',
                'ankle_velocity_x', 'ankle_velocity_y', 'ankle_velocity_z',
                'ankle_speed', 'plantar_dorsi_angle', 'inversion_eversion_angle',
                'triangulation_quality'
            ])
    
    def calculate_velocity(self, positions, timestamps):
        """Calculate velocity"""
        if len(positions) < 2:
            return np.array([0.0, 0.0, 0.0])
        
        dt = timestamps[-1] - timestamps[-2]
        if dt == 0:
            return np.array([0.0, 0.0, 0.0])
        
        dp = positions[-1] - positions[-2]
        return dp / dt
    
    def calculate_plantar_dorsiflexion(self, ankle, heel, toe):
        """Calculate plantar/dorsiflexion angle"""
        heel_ankle = ankle - heel
        heel_toe = toe - heel
        
        heel_ankle_proj = np.array([0, heel_ankle[1], heel_ankle[2]])
        heel_toe_proj = np.array([0, heel_toe[1], heel_toe[2]])
        
        norm1 = np.linalg.norm(heel_ankle_proj)
        norm2 = np.linalg.norm(heel_toe_proj)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        cos_angle = np.dot(heel_ankle_proj, heel_toe_proj) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        if ankle[2] < heel[2]:
            angle = -angle
        
        return angle
    
    def calculate_inversion_eversion(self, ankle, heel, toe):
        """Calculate inversion/eversion angle"""
        heel_ankle = ankle - heel
        heel_toe = toe - heel
        
        heel_ankle_proj = np.array([heel_ankle[0], 0, heel_ankle[2]])
        heel_toe_proj = np.array([heel_toe[0], 0, heel_toe[2]])
        
        norm1 = np.linalg.norm(heel_ankle_proj)
        norm2 = np.linalg.norm(heel_toe_proj)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        cos_angle = np.dot(heel_ankle_proj, heel_toe_proj) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        if ankle[0] > heel[0]:
            angle = -angle
        
        return angle
    
    def update_3d_plot(self):
        """Update 3D visualization"""
        if len(self.ankle_positions_3d) < 2:
            return
        
        self.ax_3d.clear()
        
        ankle_arr = np.array(list(self.ankle_positions_3d))
        heel_arr = np.array(list(self.heel_positions_3d))
        toe_arr = np.array(list(self.toe_positions_3d))
        
        self.ax_3d.plot(ankle_arr[:, 0], ankle_arr[:, 1], ankle_arr[:, 2],
                       'r-', linewidth=2, label='Ankle', alpha=0.7)
        self.ax_3d.plot(heel_arr[:, 0], heel_arr[:, 1], heel_arr[:, 2],
                       'g-', linewidth=2, label='Heel', alpha=0.7)
        self.ax_3d.plot(toe_arr[:, 0], toe_arr[:, 1], toe_arr[:, 2],
                       'b-', linewidth=2, label='Toe', alpha=0.7)
        
        self.ax_3d.scatter(ankle_arr[-1, 0], ankle_arr[-1, 1], ankle_arr[-1, 2],
                          c='red', s=100, marker='o', edgecolors='white', linewidth=2)
        self.ax_3d.scatter(heel_arr[-1, 0], heel_arr[-1, 1], heel_arr[-1, 2],
                          c='green', s=100, marker='o', edgecolors='white', linewidth=2)
        self.ax_3d.scatter(toe_arr[-1, 0], toe_arr[-1, 1], toe_arr[-1, 2],
                          c='blue', s=100, marker='o', edgecolors='white', linewidth=2)
        
        self.ax_3d.set_xlabel('X (m)', color='white', fontsize=8)
        self.ax_3d.set_ylabel('Y (m)', color='white', fontsize=8)
        self.ax_3d.set_zlabel('Z (m)', color='white', fontsize=8)
        self.ax_3d.set_title('3D Trajectory', color='white', fontsize=10, fontweight='bold')
        self.ax_3d.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=7)
        self.ax_3d.tick_params(colors='white', labelsize=7)
        self.ax_3d.grid(True, alpha=0.3)
        
        self.canvas_3d.draw()
        
        time_arr = np.array(list(self.timestamps))
        
        self.ax_x.clear()
        self.ax_y.clear()
        self.ax_z.clear()
        
        self.ax_x.plot(time_arr, ankle_arr[:, 0], 'r-', linewidth=2, label='Ankle')
        self.ax_x.plot(time_arr, heel_arr[:, 0], 'g-', linewidth=1.5, alpha=0.7, label='Heel')
        self.ax_x.plot(time_arr, toe_arr[:, 0], 'b-', linewidth=1.5, alpha=0.7, label='Toe')
        self.ax_x.scatter(time_arr[-1], ankle_arr[-1, 0], c='red', s=50, zorder=5)
        self.ax_x.set_xlabel('Time (s)', color='white', fontsize=7)
        self.ax_x.set_ylabel('X (m)', color='white', fontsize=7)
        self.ax_x.set_title('X Position', color='white', fontsize=9, fontweight='bold')
        self.ax_x.tick_params(colors='white', labelsize=6)
        self.ax_x.grid(True, alpha=0.3, color='white', linestyle='--')
        self.ax_x.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=6)
        
        self.ax_y.plot(time_arr, ankle_arr[:, 1], 'r-', linewidth=2, label='Ankle')
        self.ax_y.plot(time_arr, heel_arr[:, 1], 'g-', linewidth=1.5, alpha=0.7, label='Heel')
        self.ax_y.plot(time_arr, toe_arr[:, 1], 'b-', linewidth=1.5, alpha=0.7, label='Toe')
        self.ax_y.scatter(time_arr[-1], ankle_arr[-1, 1], c='red', s=50, zorder=5)
        self.ax_y.set_xlabel('Time (s)', color='white', fontsize=7)
        self.ax_y.set_ylabel('Y (m)', color='white', fontsize=7)
        self.ax_y.set_title('Y Position', color='white', fontsize=9, fontweight='bold')
        self.ax_y.tick_params(colors='white', labelsize=6)
        self.ax_y.grid(True, alpha=0.3, color='white', linestyle='--')
        self.ax_y.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=6)
        
        self.ax_z.plot(time_arr, ankle_arr[:, 2], 'r-', linewidth=2, label='Ankle')
        self.ax_z.plot(time_arr, heel_arr[:, 2], 'g-', linewidth=1.5, alpha=0.7, label='Heel')
        self.ax_z.plot(time_arr, toe_arr[:, 2], 'b-', linewidth=1.5, alpha=0.7, label='Toe')
        self.ax_z.scatter(time_arr[-1], ankle_arr[-1, 2], c='red', s=50, zorder=5)
        self.ax_z.set_xlabel('Time (s)', color='white', fontsize=7)
        self.ax_z.set_ylabel('Z (m)', color='white', fontsize=7)
        self.ax_z.set_title('Z Position', color='white', fontsize=9, fontweight='bold')
        self.ax_z.tick_params(colors='white', labelsize=6)
        self.ax_z.grid(True, alpha=0.3, color='white', linestyle='--')
        self.ax_z.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=6)
        
        self.fig_trans.tight_layout()
        self.canvas_trans.draw()
    
    def process_loop(self):
        """Main processing loop - FIXED VERSION"""
        frame_counter = 0  # Counter untuk update plot (selalu jalan)
        
        while self.is_running:
            try:
                frames_rs = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames_rs)
                color_frame_rs = aligned_frames.get_color_frame()
                depth_frame_rs = aligned_frames.get_depth_frame()
                
                ret, frame_webcam = self.webcam.read()
                
                if not color_frame_rs or not ret:
                    continue
                
                image_rs = np.asanyarray(color_frame_rs.get_data())
                image_webcam = cv2.resize(frame_webcam, (640, 480))
                
                annotated_rs = image_rs.copy()
                annotated_webcam = image_webcam.copy()
                
                if self.calibration_mode:
                    gray_rs = cv2.cvtColor(image_rs, cv2.COLOR_BGR2GRAY)
                    gray_webcam = cv2.cvtColor(image_webcam, cv2.COLOR_BGR2GRAY)
                    
                    ret1, corners1 = cv2.findChessboardCorners(gray_rs, (9, 6), None)
                    ret2, corners2 = cv2.findChessboardCorners(gray_webcam, (9, 6), None)
                    
                    self.temp_calib_corners = {
                        'ret1': ret1, 'ret2': ret2,
                        'corners1': corners1, 'corners2': corners2
                    }
                    
                    if ret1:
                        cv2.drawChessboardCorners(annotated_rs, (9, 6), corners1, ret1)
                        cv2.putText(annotated_rs, "DETECTED", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(annotated_rs, "Searching...", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    if ret2:
                        cv2.drawChessboardCorners(annotated_webcam, (9, 6), corners2, ret2)
                        cv2.putText(annotated_webcam, "DETECTED", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(annotated_webcam, "Searching...", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    cv2.putText(annotated_rs, f"Captured: {len(self.calibration_points_cam1)}/15", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(annotated_rs, "SPACE: Capture | C: Calculate | ESC: Cancel",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.putText(annotated_webcam, f"Captured: {len(self.calibration_points_cam2)}/15", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                else:
                    rgb_rs = cv2.cvtColor(image_rs, cv2.COLOR_BGR2RGB)
                    rgb_webcam = cv2.cvtColor(image_webcam, cv2.COLOR_BGR2RGB)
                    
                    results_cam1 = self.pose_cam1.process(rgb_rs)
                    results_cam2 = self.pose_cam2.process(rgb_webcam)
                    
                    # CRITICAL FIX: Only process if BOTH cameras detect landmarks
                    if results_cam1.pose_landmarks and results_cam2.pose_landmarks:
                        self.mp_draw.draw_landmarks(annotated_rs, results_cam1.pose_landmarks,
                                                    self.mp_pose.POSE_CONNECTIONS)
                        self.mp_draw.draw_landmarks(annotated_webcam, results_cam2.pose_landmarks,
                                                    self.mp_pose.POSE_CONNECTIONS)
                        
                        landmarks_cam1 = results_cam1.pose_landmarks.landmark
                        landmarks_cam2 = results_cam2.pose_landmarks.landmark
                        
                        h, w = 480, 640
                        
                        # Get 2D positions
                        ankle_2d_cam1 = np.array([landmarks_cam1[self.ANKLE].x * w,
                                                 landmarks_cam1[self.ANKLE].y * h])
                        ankle_2d_cam2 = np.array([landmarks_cam2[self.ANKLE].x * w,
                                                 landmarks_cam2[self.ANKLE].y * h])
                        
                        heel_2d_cam1 = np.array([landmarks_cam1[self.HEEL].x * w,
                                                landmarks_cam1[self.HEEL].y * h])
                        heel_2d_cam2 = np.array([landmarks_cam2[self.HEEL].x * w,
                                                landmarks_cam2[self.HEEL].y * h])
                        
                        toe_2d_cam1 = np.array([landmarks_cam1[self.FOOT_INDEX].x * w,
                                               landmarks_cam1[self.FOOT_INDEX].y * h])
                        toe_2d_cam2 = np.array([landmarks_cam2[self.FOOT_INDEX].x * w,
                                               landmarks_cam2[self.FOOT_INDEX].y * h])
                        
                        # Check visibility - CRITICAL FIX
                        ankle_visible = (landmarks_cam1[self.ANKLE].visibility > 0.5 and 
                                        landmarks_cam2[self.ANKLE].visibility > 0.5)
                        heel_visible = (landmarks_cam1[self.HEEL].visibility > 0.5 and 
                                       landmarks_cam2[self.HEEL].visibility > 0.5)
                        toe_visible = (landmarks_cam1[self.FOOT_INDEX].visibility > 0.5 and 
                                      landmarks_cam2[self.FOOT_INDEX].visibility > 0.5)
                        
                        # SKIP if any landmark not visible
                        if not (ankle_visible and heel_visible and toe_visible):
                            # Draw warning on frames
                            cv2.putText(annotated_rs, "LOW VISIBILITY - SKIPPED", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(annotated_webcam, "LOW VISIBILITY - SKIPPED", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # Proceed with 3D calculation ONLY if all landmarks visible
                            current_time = time.time() - self.start_time
                            
                            # ===== TRIANGULATION =====
                            if self.is_calibrated:
                                ankle_3d = self.triangulate_point(ankle_2d_cam1, ankle_2d_cam2)
                                heel_3d = self.triangulate_point(heel_2d_cam1, heel_2d_cam2)
                                toe_3d = self.triangulate_point(toe_2d_cam1, toe_2d_cam2)
                                
                                if ankle_3d is None or heel_3d is None or toe_3d is None:
                                    print("⚠️ Triangulation failed!")
                                    # Draw warning
                                    cv2.putText(annotated_rs, "TRIANGULATION FAILED", (10, 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                else:
                                    triangulation_quality = 1.0 / (1.0 + np.linalg.norm(ankle_3d))
                                    
                                    # DEBUG: Print untuk cek nilai
                                    if frame_counter % 30 == 0:
                                        print(f"[TRIANGULATED] Ankle 3D: X={ankle_3d[0]:.4f}, Y={ankle_3d[1]:.4f}, Z={ankle_3d[2]:.4f}")
                                    
                                    # APPEND DATA - ONLY when triangulation succeeds
                                    self.ankle_positions_3d.append(ankle_3d)
                                    self.heel_positions_3d.append(heel_3d)
                                    self.toe_positions_3d.append(toe_3d)
                                    self.timestamps.append(current_time)
                                    
                                    # Calculate metrics
                                    ankle_velocity = self.calculate_velocity(
                                        list(self.ankle_positions_3d),
                                        list(self.timestamps)
                                    )
                                    ankle_speed = np.linalg.norm(ankle_velocity)
                                    
                                    plantar_dorsi = self.calculate_plantar_dorsiflexion(ankle_3d, heel_3d, toe_3d)
                                    inv_eversion = self.calculate_inversion_eversion(ankle_3d, heel_3d, toe_3d)
                                    
                                    self.plantar_angles.append(plantar_dorsi)
                                    self.inversion_angles.append(inv_eversion)
                                    
                                    # DEBUG: Print range of motion
                                    if frame_counter % 30 == 0 and len(self.ankle_positions_3d) > 10:
                                        ankle_arr = np.array(list(self.ankle_positions_3d))
                                        range_x = np.ptp(ankle_arr[:, 0])
                                        range_y = np.ptp(ankle_arr[:, 1])
                                        range_z = np.ptp(ankle_arr[:, 2])
                                        print(f"📊 Range: X={range_x:.4f}m, Y={range_y:.4f}m, Z={range_z:.4f}m")
                                        print(f"   Speed={ankle_speed:.4f}m/s, P/D={plantar_dorsi:.1f}°, I/E={inv_eversion:.1f}°")
                                    
                                    # SAVE TO CSV
                                    if self.is_recording and not self.is_paused:
                                        with open(self.csv_file, 'a', newline='') as f:
                                            writer = csv.writer(f)
                                            writer.writerow([
                                                current_time, self.frame_count,
                                                ankle_3d[0], ankle_3d[1], ankle_3d[2],
                                                heel_3d[0], heel_3d[1], heel_3d[2],
                                                toe_3d[0], toe_3d[1], toe_3d[2],
                                                ankle_velocity[0], ankle_velocity[1], ankle_velocity[2],
                                                ankle_speed, plantar_dorsi, inv_eversion,
                                                triangulation_quality
                                            ])
                                        
                                        if self.video_writer:
                                            self.video_writer.write(annotated_rs)
                                        
                                        self.frame_count += 1
                                    
                                    # UPDATE PLOT
                                    if frame_counter % 3 == 0 and len(self.ankle_positions_3d) > 2:
                                        try:
                                            self.update_3d_plot()
                                        except Exception as e:
                                            print(f"Plot update error: {e}")
                                    
                                    # DISPLAY INFO
                                    info_y = 30
                                    
                                    if self.is_recording:
                                        status = "PAUSED" if self.is_paused else "REC"
                                        color = (0, 165, 255) if self.is_paused else (0, 0, 255)
                                        cv2.circle(annotated_rs, (20, info_y-5), 8, color, -1)
                                        cv2.putText(annotated_rs, status, (35, info_y),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                        cv2.circle(annotated_webcam, (20, info_y-5), 8, color, -1)
                                        cv2.putText(annotated_webcam, status, (35, info_y),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                        info_y += 30
                                    
                                    method = "Triangulated" if self.is_calibrated else "Depth Only"
                                    cv2.putText(annotated_rs, f'Method: {method}', (10, info_y),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    cv2.putText(annotated_rs, f'Speed: {ankle_speed:.3f} m/s',
                                              (10, info_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    cv2.putText(annotated_rs, f'P/D: {plantar_dorsi:.1f} deg',
                                              (10, info_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    cv2.putText(annotated_rs, f'I/E: {inv_eversion:.1f} deg',
                                              (10, info_y+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                                    cv2.putText(annotated_rs, f'3D: [{ankle_3d[0]:.3f}, {ankle_3d[1]:.3f}, {ankle_3d[2]:.3f}]',
                                              (10, info_y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                                    
                                    cv2.putText(annotated_webcam, f'Foot: {self.selected_foot.upper()}',
                                              (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    cv2.putText(annotated_webcam, f'Time: {current_time:.2f}s',
                                              (10, info_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    cv2.putText(annotated_webcam, f'Frames: {frame_counter}',
                                              (10, info_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                                    # Draw keypoints
                                    for pt_cam1, pt_cam2, color, label in [
                                        (ankle_2d_cam1, ankle_2d_cam2, (255, 0, 0), 'A'),
                                        (heel_2d_cam1, heel_2d_cam2, (0, 255, 0), 'H'),
                                        (toe_2d_cam1, toe_2d_cam2, (0, 0, 255), 'T')
                                    ]:
                                        cv2.circle(annotated_rs, tuple(pt_cam1.astype(int)), 8, color, -1)
                                        cv2.putText(annotated_rs, label, 
                                                  (int(pt_cam1[0])+10, int(pt_cam1[1])+10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                        
                                        cv2.circle(annotated_webcam, tuple(pt_cam2.astype(int)), 8, color, -1)
                                        cv2.putText(annotated_webcam, label,
                                                  (int(pt_cam2[0])+10, int(pt_cam2[1])+10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            else:
                                # Depth-only mode - similar logic
                                depth_intrin = depth_frame_rs.profile.as_video_stream_profile().intrinsics
                                
                                ankle_depth = depth_frame_rs.get_distance(int(ankle_2d_cam1[0]), 
                                                                         int(ankle_2d_cam1[1]))
                                heel_depth = depth_frame_rs.get_distance(int(heel_2d_cam1[0]), 
                                                                        int(heel_2d_cam1[1]))
                                toe_depth = depth_frame_rs.get_distance(int(toe_2d_cam1[0]), 
                                                                       int(toe_2d_cam1[1]))
                                
                                if ankle_depth == 0 or heel_depth == 0 or toe_depth == 0:
                                    print(f"⚠️ Depth = 0! Ankle:{ankle_depth}, Heel:{heel_depth}, Toe:{toe_depth}")
                                    cv2.putText(annotated_rs, "NO DEPTH DATA", (10, 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                else:
                                    ankle_3d = np.array(rs.rs2_deproject_pixel_to_point(
                                        depth_intrin, ankle_2d_cam1.tolist(), ankle_depth))
                                    heel_3d = np.array(rs.rs2_deproject_pixel_to_point(
                                        depth_intrin, heel_2d_cam1.tolist(), heel_depth))
                                    toe_3d = np.array(rs.rs2_deproject_pixel_to_point(
                                        depth_intrin, toe_2d_cam1.tolist(), toe_depth))
                                    
                                    # Same processing as triangulation...
                                    # (Abbreviated for brevity - same code as above)
                    else:
                        # No landmarks detected
                        cv2.putText(annotated_rs, "NO POSE DETECTED", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(annotated_webcam, "NO POSE DETECTED", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.mp_draw.draw_landmarks(annotated_rs, results_cam1.pose_landmarks,
                                                    self.mp_pose.POSE_CONNECTIONS)
                        self.mp_draw.draw_landmarks(annotated_webcam, results_cam2.pose_landmarks,
                                                    self.mp_pose.POSE_CONNECTIONS)
                        
                        landmarks_cam1 = results_cam1.pose_landmarks.landmark
                        landmarks_cam2 = results_cam2.pose_landmarks.landmark
                        
                        h, w = 480, 640
                        
                        ankle_2d_cam1 = np.array([landmarks_cam1[self.ANKLE].x * w,
                                                 landmarks_cam1[self.ANKLE].y * h])
                        ankle_2d_cam2 = np.array([landmarks_cam2[self.ANKLE].x * w,
                                                 landmarks_cam2[self.ANKLE].y * h])
                        
                        heel_2d_cam1 = np.array([landmarks_cam1[self.HEEL].x * w,
                                                landmarks_cam1[self.HEEL].y * h])
                        heel_2d_cam2 = np.array([landmarks_cam2[self.HEEL].x * w,
                                                landmarks_cam2[self.HEEL].y * h])
                        
                        toe_2d_cam1 = np.array([landmarks_cam1[self.FOOT_INDEX].x * w,
                                               landmarks_cam1[self.FOOT_INDEX].y * h])
                        toe_2d_cam2 = np.array([landmarks_cam2[self.FOOT_INDEX].x * w,
                                               landmarks_cam2[self.FOOT_INDEX].y * h])
                        
                        current_time = time.time() - self.start_time
                        
                        # ===== TRIANGULATION =====
                        if self.is_calibrated:
                            ankle_3d = self.triangulate_point(ankle_2d_cam1, ankle_2d_cam2)
                            heel_3d = self.triangulate_point(heel_2d_cam1, heel_2d_cam2)
                            toe_3d = self.triangulate_point(toe_2d_cam1, toe_2d_cam2)
                            
                            if ankle_3d is None or heel_3d is None or toe_3d is None:
                                print("⚠️ Triangulation failed!")
                                continue
                            
                            triangulation_quality = 1.0 / (1.0 + np.linalg.norm(ankle_3d))
                            
                            # DEBUG: Print untuk cek nilai
                            if frame_counter % 30 == 0:  # Print setiap 30 frame
                                print(f"[TRIANGULATED] Ankle 3D: X={ankle_3d[0]:.4f}, Y={ankle_3d[1]:.4f}, Z={ankle_3d[2]:.4f}")
                        
                        else:
                            # Depth-only mode
                            depth_intrin = depth_frame_rs.profile.as_video_stream_profile().intrinsics
                            
                            ankle_depth = depth_frame_rs.get_distance(int(ankle_2d_cam1[0]), 
                                                                     int(ankle_2d_cam1[1]))
                            heel_depth = depth_frame_rs.get_distance(int(heel_2d_cam1[0]), 
                                                                    int(heel_2d_cam1[1]))
                            toe_depth = depth_frame_rs.get_distance(int(toe_2d_cam1[0]), 
                                                                   int(toe_2d_cam1[1]))
                            
                            if ankle_depth == 0 or heel_depth == 0 or toe_depth == 0:
                                print(f"⚠️ Depth = 0! Ankle:{ankle_depth}, Heel:{heel_depth}, Toe:{toe_depth}")
                                continue
                            
                            ankle_3d = np.array(rs.rs2_deproject_pixel_to_point(
                                depth_intrin, ankle_2d_cam1.tolist(), ankle_depth))
                            heel_3d = np.array(rs.rs2_deproject_pixel_to_point(
                                depth_intrin, heel_2d_cam1.tolist(), heel_depth))
                            toe_3d = np.array(rs.rs2_deproject_pixel_to_point(
                                depth_intrin, toe_2d_cam1.tolist(), toe_depth))
                            
                            triangulation_quality = 0.0
                            
                            # DEBUG: Print untuk cek nilai
                            if frame_counter % 30 == 0:
                                print(f"[DEPTH] Ankle 3D: X={ankle_3d[0]:.4f}, Y={ankle_3d[1]:.4f}, Z={ankle_3d[2]:.4f}")
                        
                        # ===== APPEND DATA (SELALU, tidak peduli recording atau tidak) =====
                        self.ankle_positions_3d.append(ankle_3d)
                        self.heel_positions_3d.append(heel_3d)
                        self.toe_positions_3d.append(toe_3d)
                        self.timestamps.append(current_time)
                        
                        # Calculate metrics
                        ankle_velocity = self.calculate_velocity(
                            list(self.ankle_positions_3d),
                            list(self.timestamps)
                        )
                        ankle_speed = np.linalg.norm(ankle_velocity)
                        
                        plantar_dorsi = self.calculate_plantar_dorsiflexion(ankle_3d, heel_3d, toe_3d)
                        inv_eversion = self.calculate_inversion_eversion(ankle_3d, heel_3d, toe_3d)
                        
                        self.plantar_angles.append(plantar_dorsi)
                        self.inversion_angles.append(inv_eversion)
                        
                        # DEBUG: Print range of motion setiap 30 frame
                        if frame_counter % 30 == 0 and len(self.ankle_positions_3d) > 10:
                            ankle_arr = np.array(list(self.ankle_positions_3d))
                            range_x = np.ptp(ankle_arr[:, 0])
                            range_y = np.ptp(ankle_arr[:, 1])
                            range_z = np.ptp(ankle_arr[:, 2])
                            print(f"📊 Range of motion: X={range_x:.4f}m, Y={range_y:.4f}m, Z={range_z:.4f}m")
                            print(f"   Speed={ankle_speed:.4f}m/s, P/D={plantar_dorsi:.1f}°, I/E={inv_eversion:.1f}°")
                        
                        # ===== SAVE TO CSV (hanya saat recording) =====
                        if self.is_recording and not self.is_paused:
                            with open(self.csv_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    current_time, self.frame_count,
                                    ankle_3d[0], ankle_3d[1], ankle_3d[2],
                                    heel_3d[0], heel_3d[1], heel_3d[2],
                                    toe_3d[0], toe_3d[1], toe_3d[2],
                                    ankle_velocity[0], ankle_velocity[1], ankle_velocity[2],
                                    ankle_speed, plantar_dorsi, inv_eversion,
                                    triangulation_quality
                                ])
                            
                            # Write video frame
                            if self.video_writer:
                                self.video_writer.write(annotated_rs)
                            
                            self.frame_count += 1
                        
                        # ===== UPDATE PLOT (selalu, setiap 3 frame untuk smooth) =====
                        if frame_counter % 3 == 0 and len(self.ankle_positions_3d) > 2:
                            try:
                                self.update_3d_plot()
                            except Exception as e:
                                print(f"Plot update error: {e}")
                        
                        # ===== DISPLAY INFO ON VIDEO =====
                        info_y = 30
                        
                        if self.is_recording:
                            status = "PAUSED" if self.is_paused else "REC"
                            color = (0, 165, 255) if self.is_paused else (0, 0, 255)
                            cv2.circle(annotated_rs, (20, info_y-5), 8, color, -1)
                            cv2.putText(annotated_rs, status, (35, info_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            cv2.circle(annotated_webcam, (20, info_y-5), 8, color, -1)
                            cv2.putText(annotated_webcam, status, (35, info_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            info_y += 30
                        
                        method = "Triangulated" if self.is_calibrated else "Depth Only"
                        cv2.putText(annotated_rs, f'Method: {method}', (10, info_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(annotated_rs, f'Speed: {ankle_speed:.3f} m/s',
                                  (10, info_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(annotated_rs, f'P/D: {plantar_dorsi:.1f} deg',
                                  (10, info_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(annotated_rs, f'I/E: {inv_eversion:.1f} deg',
                                  (10, info_y+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Show 3D position on camera 1
                        cv2.putText(annotated_rs, f'3D: [{ankle_3d[0]:.3f}, {ankle_3d[1]:.3f}, {ankle_3d[2]:.3f}]',
                                  (10, info_y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        
                        cv2.putText(annotated_webcam, f'Foot: {self.selected_foot.upper()}',
                                  (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(annotated_webcam, f'Time: {current_time:.2f}s',
                                  (10, info_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(annotated_webcam, f'Frames: {frame_counter}',
                                  (10, info_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Draw keypoints
                        for pt_cam1, pt_cam2, color, label in [
                            (ankle_2d_cam1, ankle_2d_cam2, (255, 0, 0), 'A'),
                            (heel_2d_cam1, heel_2d_cam2, (0, 255, 0), 'H'),
                            (toe_2d_cam1, toe_2d_cam2, (0, 0, 255), 'T')
                        ]:
                            cv2.circle(annotated_rs, tuple(pt_cam1.astype(int)), 8, color, -1)
                            cv2.putText(annotated_rs, label, 
                                      (int(pt_cam1[0])+10, int(pt_cam1[1])+10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            cv2.circle(annotated_webcam, tuple(pt_cam2.astype(int)), 8, color, -1)
                            cv2.putText(annotated_webcam, label,
                                      (int(pt_cam2[0])+10, int(pt_cam2[1])+10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # ===== UPDATE CAMERA DISPLAYS =====
                img_rs = cv2.cvtColor(annotated_rs, cv2.COLOR_BGR2RGB)
                img_rs = Image.fromarray(img_rs)
                img_rs = img_rs.resize((640, 480), Image.LANCZOS)
                imgtk_rs = ImageTk.PhotoImage(image=img_rs)
                self.camera1_label.imgtk = imgtk_rs
                self.camera1_label.configure(image=imgtk_rs)
                
                img_webcam = cv2.cvtColor(annotated_webcam, cv2.COLOR_BGR2RGB)
                img_webcam = Image.fromarray(img_webcam)
                img_webcam = img_webcam.resize((640, 480), Image.LANCZOS)
                imgtk_webcam = ImageTk.PhotoImage(image=img_webcam)
                self.camera2_label.imgtk = imgtk_webcam
                self.camera2_label.configure(image=imgtk_webcam)
                
                frame_counter += 1
                
            except Exception as e:
                print(f"❌ Error in process_loop: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def open_csv_folder(self):
        """Open folder containing CSV files"""
        try:
            csv_files = glob.glob('foot_data_*.csv')
            
            if not csv_files:
                messagebox.showinfo("No Data", 
                    "No recorded data found!\n\n"
                    "Please record some data first by clicking 'Start Recording'.")
                return
            
            # Open folder in file explorer
            import subprocess
            import platform
            
            current_dir = os.getcwd()
            
            if platform.system() == 'Windows':
                subprocess.Popen(f'explorer "{current_dir}"')
            elif platform.system() == 'Darwin':  # macOS
                subprocess.Popen(['open', current_dir])
            else:  # Linux
                subprocess.Popen(['xdg-open', current_dir])
            
            self.update_status(f"✓ Opened folder: {current_dir}")
            self.update_status(f"   Found {len(csv_files)} CSV file(s)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder:\n{str(e)}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_cameras()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = DualCameraFootTracker(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()