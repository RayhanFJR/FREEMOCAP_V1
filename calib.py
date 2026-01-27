import cv2
import numpy as np
import pyrealsense2 as rs
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
import threading
from datetime import datetime

class CameraCalibrationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Camera Calibration Tool - Checkerboard & ChArUco")
        self.root.geometry("1600x900")
        self.root.configure(bg='#2b2b2b')
        
        # Camera setup
        self.pipeline = None
        self.webcam = None
        self.webcam_index = 0
        
        # State
        self.is_running = False
        self.calibration_mode = False
        self.pattern_type = "checkerboard"
        
        # Pattern settings - Checkerboard
        self.checker_rows = 6
        self.checker_cols = 9
        self.checker_square_size = 0.025
        
        # Pattern settings - ChArUco
        self.charuco_rows = 5
        self.charuco_cols = 7
        self.charuco_square_size = 0.04
        self.charuco_marker_size = 0.03
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        
        # Calibration data
        self.objpoints = []
        self.imgpoints_cam1 = []
        self.imgpoints_cam2 = []
        
        # Detection state
        self.last_detection = {'detected': False, 'objp': None, 'corners1': None, 'corners2': None}
        
        # Results
        self.camera_matrix_1 = None
        self.dist_coeffs_1 = None
        self.camera_matrix_2 = None
        self.dist_coeffs_2 = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        
        self.setup_gui()
    
    def setup_gui(self):
        # Top control panel
        control_panel = tk.Frame(self.root, bg='#1e1e1e', height=150)
        control_panel.pack(fill=tk.X, padx=10, pady=(10, 5))
        control_panel.pack_propagate(False)
        
        # Pattern selection
        pattern_frame = tk.LabelFrame(control_panel, text="Calibration Pattern", 
                                     bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        pattern_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        type_frame = tk.Frame(pattern_frame, bg='#1e1e1e')
        type_frame.pack(pady=5)
        
        tk.Label(type_frame, text="Pattern Type:", bg='#1e1e1e', 
                fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        self.pattern_var = tk.StringVar(value="checkerboard")
        tk.Radiobutton(type_frame, text="Checkerboard", variable=self.pattern_var,
                      value="checkerboard", bg='#1e1e1e', fg='white',
                      selectcolor='#3e3e3e', command=self.on_pattern_change).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(type_frame, text="ChArUco", variable=self.pattern_var,
                      value="charuco", bg='#1e1e1e', fg='white',
                      selectcolor='#3e3e3e', command=self.on_pattern_change).pack(side=tk.LEFT, padx=5)
        
        settings_frame = tk.Frame(pattern_frame, bg='#1e1e1e')
        settings_frame.pack(pady=5)
        
        # Checkerboard settings
        self.checker_frame = tk.Frame(settings_frame, bg='#1e1e1e')
        self.checker_frame.pack()
        
        tk.Label(self.checker_frame, text="Rows:", bg='#1e1e1e', fg='white').grid(row=0, column=0, padx=3)
        self.checker_rows_var = tk.StringVar(value="6")
        tk.Entry(self.checker_frame, textvariable=self.checker_rows_var, width=5).grid(row=0, column=1, padx=3)
        
        tk.Label(self.checker_frame, text="Cols:", bg='#1e1e1e', fg='white').grid(row=0, column=2, padx=3)
        self.checker_cols_var = tk.StringVar(value="9")
        tk.Entry(self.checker_frame, textvariable=self.checker_cols_var, width=5).grid(row=0, column=3, padx=3)
        
        tk.Label(self.checker_frame, text="Square(mm):", bg='#1e1e1e', fg='white').grid(row=0, column=4, padx=3)
        self.checker_size_var = tk.StringVar(value="25")
        tk.Entry(self.checker_frame, textvariable=self.checker_size_var, width=5).grid(row=0, column=5, padx=3)
        
        # ChArUco settings
        self.charuco_frame = tk.Frame(settings_frame, bg='#1e1e1e')
        
        tk.Label(self.charuco_frame, text="Rows:", bg='#1e1e1e', fg='white').grid(row=0, column=0, padx=3)
        self.charuco_rows_var = tk.StringVar(value="5")
        tk.Entry(self.charuco_frame, textvariable=self.charuco_rows_var, width=5).grid(row=0, column=1, padx=3)
        
        tk.Label(self.charuco_frame, text="Cols:", bg='#1e1e1e', fg='white').grid(row=0, column=2, padx=3)
        self.charuco_cols_var = tk.StringVar(value="7")
        tk.Entry(self.charuco_frame, textvariable=self.charuco_cols_var, width=5).grid(row=0, column=3, padx=3)
        
        tk.Label(self.charuco_frame, text="Square(mm):", bg='#1e1e1e', fg='white').grid(row=1, column=0, padx=3)
        self.charuco_square_var = tk.StringVar(value="40")
        tk.Entry(self.charuco_frame, textvariable=self.charuco_square_var, width=5).grid(row=1, column=1, padx=3)
        
        tk.Label(self.charuco_frame, text="Marker(mm):", bg='#1e1e1e', fg='white').grid(row=1, column=2, padx=3)
        self.charuco_marker_var = tk.StringVar(value="30")
        tk.Entry(self.charuco_frame, textvariable=self.charuco_marker_var, width=5).grid(row=1, column=3, padx=3)
        
        tk.Button(pattern_frame, text="Generate Pattern", command=self.generate_pattern, 
                 bg='#9C27B0', fg='white', font=('Arial', 9, 'bold')).pack(pady=5)
        
        # Camera settings
        camera_frame = tk.LabelFrame(control_panel, text="Camera Settings", 
                                    bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Label(camera_frame, text="Webcam Index:", bg='#1e1e1e', fg='white').pack(pady=5)
        self.webcam_var = tk.StringVar(value="0")
        tk.Entry(camera_frame, textvariable=self.webcam_var, width=10).pack(pady=5)
        
        # Control buttons
        button_frame = tk.LabelFrame(control_panel, text="Controls", 
                                    bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        button_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.start_btn = tk.Button(button_frame, text="Start Cameras", command=self.start_cameras, 
                                   bg='#4CAF50', fg='white', font=('Arial', 9, 'bold'), width=15, height=2)
        self.start_btn.pack(pady=3)
        
        self.calibrate_btn = tk.Button(button_frame, text="Start Calibration", command=self.start_calibration, 
                                       bg='#2196F3', fg='white', font=('Arial', 9, 'bold'), 
                                       width=15, height=2, state=tk.DISABLED)
        self.calibrate_btn.pack(pady=3)
        
        self.stop_btn = tk.Button(button_frame, text="Stop", command=self.stop_cameras, 
                                  bg='#f44336', fg='white', font=('Arial', 9, 'bold'), 
                                  width=15, height=2, state=tk.DISABLED)
        self.stop_btn.pack(pady=3)
        
        # Camera feeds
        camera_feed_frame = tk.Frame(self.root, bg='#2b2b2b')
        camera_feed_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        cam1_frame = tk.LabelFrame(camera_feed_frame, text="Camera 1 - RealSense", 
                                   bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        cam1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.camera1_label = tk.Label(cam1_frame, bg='black')
        self.camera1_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        cam2_frame = tk.LabelFrame(camera_feed_frame, text="Camera 2 - Webcam", 
                                   bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        cam2_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.camera2_label = tk.Label(cam2_frame, bg='black')
        self.camera2_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status panel
        status_frame = tk.LabelFrame(self.root, text="Status & Results", 
                                    bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'), height=200)
        status_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        status_frame.pack_propagate(False)
        
        log_frame = tk.Frame(status_frame, bg='#1e1e1e')
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.status_text = tk.Text(log_frame, height=10, bg='#2b2b2b', fg='#00ff00', 
                                  font=('Courier', 8), state=tk.DISABLED, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(log_frame, command=self.status_text.yview)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        info_frame = tk.Frame(status_frame, bg='#1e1e1e', width=400)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        info_frame.pack_propagate(False)
        
        self.calib_info = tk.Text(info_frame, bg='#2b2b2b', fg='cyan', 
                                 font=('Courier', 9), state=tk.DISABLED)
        self.calib_info.pack(fill=tk.BOTH, expand=True)
        
        # Keyboard bindings
        self.root.bind('<space>', self.on_space_pressed)
        self.root.bind('c', self.on_c_pressed)
        self.root.bind('C', self.on_c_pressed)
        self.root.bind('<Escape>', self.on_esc_pressed)
        
        self.update_calibration_info()
    
    def on_space_pressed(self, event):
        """Handle space key press for capturing frames"""
        if self.calibration_mode and self.last_detection['detected']:
            self.objpoints.append(self.last_detection['objp'])
            self.imgpoints_cam1.append(self.last_detection['corners1'])
            self.imgpoints_cam2.append(self.last_detection['corners2'])
            self.update_status(f"✓ Captured frame {len(self.objpoints)}")
            self.update_calibration_info()
    
    def on_c_pressed(self, event):
        """Handle 'C' key press for calculating calibration"""
        if self.calibration_mode and len(self.objpoints) > 0:
            self.calibration_mode = False
            self.perform_calibration()
    
    def on_esc_pressed(self, event):
        """Handle ESC key press for canceling calibration"""
        if self.calibration_mode:
            self.calibration_mode = False
            self.update_status("✗ Calibration cancelled")
    
    def on_pattern_change(self):
        self.pattern_type = self.pattern_var.get()
        if self.pattern_type == "checkerboard":
            self.charuco_frame.pack_forget()
            self.checker_frame.pack()
        else:
            self.checker_frame.pack_forget()
            self.charuco_frame.pack()
        self.update_status(f"Pattern changed to: {self.pattern_type.upper()}")
    
    def generate_pattern(self):
        try:
            if self.pattern_type == "checkerboard":
                rows = int(self.checker_rows_var.get())
                cols = int(self.checker_cols_var.get())
                square_size = int(self.checker_size_var.get())
                
                px = 100
                img = np.ones((rows * px, cols * px), dtype=np.uint8) * 255
                
                for i in range(rows):
                    for j in range(cols):
                        if (i + j) % 2 == 0:
                            img[i*px:(i+1)*px, j*px:(j+1)*px] = 0
                
                filename = f"checkerboard_{rows}x{cols}_{square_size}mm.png"
            else:
                rows = int(self.charuco_rows_var.get())
                cols = int(self.charuco_cols_var.get())
                square_size = int(self.charuco_square_var.get())
                marker_size = int(self.charuco_marker_var.get())
                
                board = cv2.aruco.CharucoBoard(
                    (cols, rows),
                    square_size / 1000.0,
                    marker_size / 1000.0,
                    self.aruco_dict
                )
                
                img = board.generateImage((cols * 100, rows * 100))
                filename = f"charuco_{rows}x{cols}_{square_size}mm_{marker_size}mm.png"
            
            cv2.imwrite(filename, img)
            self.update_status(f"✓ Pattern saved: {filename}")
            messagebox.showinfo("Success", f"Pattern saved:\n{filename}\n\nPrint this pattern and use it for calibration.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate pattern:\n{str(e)}")
    
    def update_status(self, message):
        self.status_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
    
    def update_calibration_info(self):
        self.calib_info.config(state=tk.NORMAL)
        self.calib_info.delete('1.0', tk.END)
        
        info = "CALIBRATION PROGRESS\n"
        info += "=" * 40 + "\n\n"
        info += f"Pattern: {self.pattern_type.upper()}\n"
        info += f"Captured Frames: {len(self.objpoints)}\n"
        info += f"Status: {'✓ READY' if len(self.objpoints) >= 10 else '⚠ Need 10+ frames'}\n\n"
        
        if self.camera_matrix_1 is not None:
            info += "✓ CALIBRATION COMPLETE\n"
            info += "-" * 40 + "\n"
            info += f"Camera 1 Focal Length:\n"
            info += f"  fx = {self.camera_matrix_1[0,0]:.2f} px\n"
            info += f"  fy = {self.camera_matrix_1[1,1]:.2f} px\n\n"
            info += f"Camera 2 Focal Length:\n"
            info += f"  fx = {self.camera_matrix_2[0,0]:.2f} px\n"
            info += f"  fy = {self.camera_matrix_2[1,1]:.2f} px\n\n"
            info += f"Stereo Baseline:\n"
            info += f"  {np.linalg.norm(self.T)*100:.2f} cm\n"
        
        self.calib_info.insert('1.0', info)
        self.calib_info.config(state=tk.DISABLED)
    
    def start_cameras(self):
        try:
            # Start RealSense
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            
            # Start Webcam
            self.webcam_index = int(self.webcam_var.get())
            self.webcam = cv2.VideoCapture(self.webcam_index)
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self.webcam.isOpened():
                raise Exception(f"Cannot open webcam at index {self.webcam_index}")
            
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.calibrate_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.update_status("✓ Cameras started successfully")
            
            # Start processing thread
            threading.Thread(target=self.process_loop, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start cameras:\n{str(e)}")
            self.cleanup_cameras()
    
    def stop_cameras(self):
        self.is_running = False
        self.calibration_mode = False
        self.cleanup_cameras()
        self.start_btn.config(state=tk.NORMAL)
        self.calibrate_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status("✓ Cameras stopped")
    
    def cleanup_cameras(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None
        if self.webcam:
            self.webcam.release()
            self.webcam = None
    
    def start_calibration(self):
        # Update pattern settings
        if self.pattern_type == "checkerboard":
            self.checker_rows = int(self.checker_rows_var.get())
            self.checker_cols = int(self.checker_cols_var.get())
            self.checker_square_size = float(self.checker_size_var.get()) / 1000.0
        else:
            self.charuco_rows = int(self.charuco_rows_var.get())
            self.charuco_cols = int(self.charuco_cols_var.get())
            self.charuco_square_size = float(self.charuco_square_var.get()) / 1000.0
            self.charuco_marker_size = float(self.charuco_marker_var.get()) / 1000.0
        
        # Reset calibration data
        self.calibration_mode = True
        self.objpoints = []
        self.imgpoints_cam1 = []
        self.imgpoints_cam2 = []
        
        msg = f"""CALIBRATION MODE STARTED

Pattern: {self.pattern_type.upper()}

Controls:
  SPACE - Capture current frame
  C     - Calculate calibration
  ESC   - Cancel calibration

Instructions:
1. Show the calibration pattern to both cameras
2. When "DETECTED" appears, press SPACE
3. Capture 15-20 frames from different angles
4. Press C to calculate calibration

TIP: Vary the distance and angle of the pattern!"""
        
        self.update_status(f"▶ Calibration mode started - {self.pattern_type.upper()}")
        self.update_calibration_info()
        messagebox.showinfo("Calibration Started", msg)
    
    def detect_pattern(self, gray1, gray2):
        if self.pattern_type == "checkerboard":
            return self.detect_checkerboard(gray1, gray2)
        else:
            return self.detect_charuco(gray1, gray2)
    
    def detect_checkerboard(self, gray1, gray2):
        pattern_size = (self.checker_cols, self.checker_rows)
        ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)
        
        if ret1 and ret2:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            
            objp = np.zeros((self.checker_rows * self.checker_cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.checker_cols, 0:self.checker_rows].T.reshape(-1, 2)
            objp *= self.checker_square_size
            
            return True, objp, corners1, corners2
        return False, None, None, None
    
    def detect_charuco(self, gray1, gray2):
        board = cv2.aruco.CharucoBoard(
            (self.charuco_cols, self.charuco_rows),
            self.charuco_square_size,
            self.charuco_marker_size,
            self.aruco_dict
        )
        
        params = cv2.aruco.DetectorParameters()
        charuco_params = cv2.aruco.CharucoParameters()
        detector = cv2.aruco.CharucoDetector(board, charuco_params, params)
        
        corners1, ids1, _, _ = detector.detectBoard(gray1)
        corners2, ids2, _, _ = detector.detectBoard(gray2)
        
        if corners1 is not None and corners2 is not None and len(corners1) > 4 and len(corners2) > 4:
            common_ids = np.intersect1d(ids1, ids2)
            if len(common_ids) < 4:
                return False, None, None, None
            
            mask1 = np.isin(ids1.flatten(), common_ids)
            mask2 = np.isin(ids2.flatten(), common_ids)
            
            matched_corners1 = corners1[mask1]
            matched_corners2 = corners2[mask2]
            matched_ids = ids1[mask1]
            
            objp = board.getChessboardCorners()[matched_ids.flatten()]
            return True, objp, matched_corners1, matched_corners2
        return False, None, None, None
    
    def perform_calibration(self):
        if len(self.objpoints) < 10:
            messagebox.showwarning("Insufficient Data", 
                f"Need at least 10 frames.\nCurrently captured: {len(self.objpoints)}")
            return
        
        try:
            self.update_status("⏳ Calculating calibration parameters...")
            
            # Individual camera calibration
            ret1, mtx1, dist1, _, _ = cv2.calibrateCamera(
                self.objpoints, self.imgpoints_cam1, (640, 480), None, None)
            ret2, mtx2, dist2, _, _ = cv2.calibrateCamera(
                self.objpoints, self.imgpoints_cam2, (640, 480), None, None)
            
            self.update_status(f"  Camera 1 RMS error: {ret1:.4f}")
            self.update_status(f"  Camera 2 RMS error: {ret2:.4f}")
            
            # Stereo calibration
            flags = cv2.CALIB_FIX_INTRINSIC
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
            
            ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
                self.objpoints, self.imgpoints_cam1, self.imgpoints_cam2,
                mtx1, dist1, mtx2, dist2, (640, 480), criteria=criteria, flags=flags)
            
            # Store results
            self.camera_matrix_1 = mtx1
            self.dist_coeffs_1 = dist1
            self.camera_matrix_2 = mtx2
            self.dist_coeffs_2 = dist2
            self.R = R
            self.T = T
            self.E = E
            self.F = F
            
            baseline_cm = np.linalg.norm(T) * 100
            
            self.update_status(f"✓ Stereo calibration RMS error: {ret:.4f}")
            self.update_status(f"✓ Baseline distance: {baseline_cm:.2f} cm")
            self.update_calibration_info()
            self.save_calibration()
            
            result_msg = f"""CALIBRATION SUCCESSFUL!

Stereo RMS Error: {ret:.4f}
Baseline Distance: {baseline_cm:.2f} cm

Camera 1 Focal Length:
  fx = {mtx1[0,0]:.2f} px
  fy = {mtx1[1,1]:.2f} px

Camera 2 Focal Length:
  fx = {mtx2[0,0]:.2f} px
  fy = {mtx2[1,1]:.2f} px

Results saved to: camera_calibration.json"""
            
            messagebox.showinfo("Calibration Complete", result_msg)
        except Exception as e:
            messagebox.showerror("Calibration Failed", f"Error during calibration:\n{str(e)}")
            self.update_status(f"✗ Calibration failed: {str(e)}")
    
    def save_calibration(self):
        data = {
            'pattern_type': self.pattern_type,
            'camera_matrix_1': self.camera_matrix_1.tolist(),
            'dist_coeffs_1': self.dist_coeffs_1.tolist(),
            'camera_matrix_2': self.camera_matrix_2.tolist(),
            'dist_coeffs_2': self.dist_coeffs_2.tolist(),
            'R': self.R.tolist(),
            'T': self.T.tolist(),
            'E': self.E.tolist(),
            'F': self.F.tolist(),
            'baseline': float(np.linalg.norm(self.T)),
            'image_size': [640, 480],
            'frames_used': len(self.objpoints),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        filename = "camera_calibration.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        self.update_status(f"✓ Calibration saved to {filename}")
    
    def process_loop(self):
        """Main processing loop for camera feeds"""
        while self.is_running:
            try:
                # Get frames from both cameras
                frames_rs = self.pipeline.wait_for_frames()
                color_rs = frames_rs.get_color_frame()
                ret_web, frame_web = self.webcam.read()
                
                if not color_rs or not ret_web:
                    continue
                
                # Convert to numpy arrays
                img_rs = np.asanyarray(color_rs.get_data())
                img_web = cv2.resize(frame_web, (640, 480))
                
                # Create display copies
                display_rs = img_rs.copy()
                display_web = img_web.copy()
                
                # Pattern detection in calibration mode
                if self.calibration_mode:
                    gray1 = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(img_web, cv2.COLOR_BGR2GRAY)
                    
                    # Detect pattern
                    detected, objp, corners1, corners2 = self.detect_pattern(gray1, gray2)
                    
                    # Store detection result
                    self.last_detection = {
                        'detected': detected,
                        'objp': objp,
                        'corners1': corners1,
                        'corners2': corners2
                    }
                    
                    # Visualize detection
                    if detected:
                        if self.pattern_type == "checkerboard":
                            cv2.drawChessboardCorners(display_rs, (self.checker_cols, self.checker_rows), 
                                                     corners1, True)
                            cv2.drawChessboardCorners(display_web, (self.checker_cols, self.checker_rows), 
                                                     corners2, True)
                        else:
                            for corner in corners1:
                                cv2.circle(display_rs, tuple(corner[0].astype(int)), 5, (0, 255, 0), -1)
                            for corner in corners2:
                                cv2.circle(display_web, tuple(corner[0].astype(int)), 5, (0, 255, 0), -1)
                        
                        cv2.putText(display_rs, "DETECTED - Press SPACE", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_web, "DETECTED - Press SPACE", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_rs, "Searching for pattern...", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        cv2.putText(display_web, "Searching for pattern...", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    # Show capture count
                    cv2.putText(display_rs, f"Captured: {len(self.objpoints)}/20", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display_web, f"Captured: {len(self.objpoints)}/20", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display_rs, "C: Calculate | ESC: Cancel", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Convert and display frames
                img_rs_rgb = cv2.cvtColor(display_rs, cv2.COLOR_BGR2RGB)
                img_rs_pil = Image.fromarray(img_rs_rgb)
                img_rs_pil = img_rs_pil.resize((640, 480), Image.LANCZOS)
                imgtk_rs = ImageTk.PhotoImage(image=img_rs_pil)
                self.camera1_label.imgtk = imgtk_rs
                self.camera1_label.configure(image=imgtk_rs)
                
                img_web_rgb = cv2.cvtColor(display_web, cv2.COLOR_BGR2RGB)
                img_web_pil = Image.fromarray(img_web_rgb)
                img_web_pil = img_web_pil.resize((640, 480), Image.LANCZOS)
                imgtk_web = ImageTk.PhotoImage(image=img_web_pil)
                self.camera2_label.imgtk = imgtk_web
                self.camera2_label.configure(image=imgtk_web)
                
            except Exception as e:
                print(f"Error in process loop: {e}")
                continue
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup_cameras()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = CameraCalibrationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()