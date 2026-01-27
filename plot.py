import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datetime import datetime
import cv2
from PIL import Image, ImageTk
from scipy.signal import find_peaks, butter, filtfilt
from filterpy.kalman import KalmanFilter

class DataFilter:
    """Class untuk filtering data motion capture"""
    
    @staticmethod
    def remove_outliers(data, threshold=3):
        """Buang outliers menggunakan Z-score"""
        data_array = np.array(data)
        mean = np.nanmean(data_array)
        std = np.nanstd(data_array)
        
        if std == 0:
            return data_array
        
        z_scores = np.abs((data_array - mean) / std)
        filtered = np.where(z_scores < threshold, data_array, np.nan)
        
        return filtered
    
    @staticmethod
    def interpolate_nans(data):
        """Interpolasi nilai NaN"""
        data_array = np.array(data)
        mask = np.isnan(data_array)
        
        if not mask.any():
            return data_array
        
        indices = np.arange(len(data_array))
        data_array[mask] = np.interp(indices[mask], indices[~mask], data_array[~mask])
        
        return data_array
    
    @staticmethod
    def kalman_filter_1d(data, R=5, Q=0.1):
        """
        Kalman filter untuk 1D data
        
        Parameters:
        - R: Measurement noise (makin besar = lebih smooth tapi slower response)
        - Q: Process noise (makin kecil = lebih percaya model)
        """
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # State transition matrix (position, velocity)
        kf.F = np.array([[1., 1.],
                         [0., 1.]])
        
        # Measurement function
        kf.H = np.array([[1., 0.]])
        
        # Covariance matrices
        kf.P *= 1000.
        kf.R = R
        kf.Q = np.array([[Q, 0.],
                         [0., Q]])
        
        # Initialize state
        kf.x = np.array([[data[0]], [0.]])
        
        # Filter
        filtered = []
        for z in data:
            kf.predict()
            if not np.isnan(z):
                kf.update(z)
            filtered.append(kf.x[0, 0])
        
        return np.array(filtered)
    
    @staticmethod
    def butterworth_filter(data, cutoff=5, fs=30, order=4):
        """
        Low-pass Butterworth filter
        
        Parameters:
        - cutoff: Cutoff frequency (Hz) - human gait ~2-5 Hz
        - fs: Sampling frequency (Hz)
        - order: Filter order (makin tinggi = lebih tajam cutoff)
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        
        if normal_cutoff >= 1:
            return data
        
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # Use filtfilt for zero phase delay
        filtered = filtfilt(b, a, data)
        
        return filtered
    
    @staticmethod
    def apply_full_pipeline(raw_data, fps=30, outlier_threshold=3, 
                           kalman_R=5, kalman_Q=0.1, 
                           butter_cutoff=5, butter_order=4):
        """
        Complete filtering pipeline
        
        Returns:
        - filtered_data: Cleaned data
        - stats: Dictionary with statistics
        """
        # Step 1: Remove outliers
        data = DataFilter.remove_outliers(raw_data, threshold=outlier_threshold)
        outliers_removed = np.sum(np.isnan(data))
        
        # Step 2: Interpolate gaps
        data = DataFilter.interpolate_nans(data)
        
        # Step 3: Kalman filter
        data = DataFilter.kalman_filter_1d(data, R=kalman_R, Q=kalman_Q)
        
        # Step 4: Butterworth low-pass
        data = DataFilter.butterworth_filter(data, cutoff=butter_cutoff, 
                                            fs=fps, order=butter_order)
        
        # Calculate statistics
        raw_array = np.array(raw_data)
        noise_reduction = np.std(raw_array) - np.std(data)
        
        stats = {
            'outliers_removed': outliers_removed,
            'noise_reduction': noise_reduction,
            'raw_std': np.std(raw_array),
            'filtered_std': np.std(data)
        }
        
        return data, stats
    
    @staticmethod
    def clean_motion_data(raw_data, fps=30, outlier_sigma=3, 
                        kalman_R=2, kalman_Q=0.05, cutoff_hz=3):
        """
        Simplified wrapper untuk apply_full_pipeline
        
        Parameters:
        - outlier_sigma: Z-score threshold (default 3 = 99.7%)
        - kalman_R: Measurement noise (1-10, smaller = smoother)
        - kalman_Q: Process noise (0.01-0.5, smaller = trust model more)
        - cutoff_hz: Butterworth cutoff frequency (1-8 Hz for gait)
        """
        filtered_data, stats = DataFilter.apply_full_pipeline(
            raw_data,
            fps=fps,
            outlier_threshold=outlier_sigma,
            kalman_R=kalman_R,
            kalman_Q=kalman_Q,
            butter_cutoff=cutoff_hz,
            butter_order=4
        )
        
        return filtered_data

class FootDataAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Foot Movement Data Analyzer with Advanced Filtering")
        self.root.geometry("1800x1000")
        self.root.configure(bg='#2b2b2b')
        
        self.df = None
        self.df_filtered = None  # Filtered data
        self.current_file = None
        self.video_file = None
        self.video_cap = None
        self.current_frame_idx = 0
        self.is_playing = False
        self.video_fps = 30
        self.is_dual_camera = False
        
        # Filter settings
        self.use_filtered_data = tk.BooleanVar(value=False)
        self.outlier_threshold = tk.DoubleVar(value=3.0)
        self.kalman_R = tk.DoubleVar(value=5.0)
        self.kalman_Q = tk.DoubleVar(value=0.1)
        self.butter_cutoff = tk.DoubleVar(value=5.0)
        self.butter_order = tk.IntVar(value=4)
        
        # Reference trajectory
        self.reference_trajectory = None
        self.reference_file = None
        
        # Gait cycle detection
        self.gait_cycles = []
        self.current_cycle_idx = 0
        
        # GUI components
        self.frame_slider = None
        self.traj_fig = None
        self.traj_canvas = None
        self.gait_fig = None
        self.gait_canvas = None
        self.video_canvas = None
        self.play_button = None
        self.frame_label = None
        self.file_label = None
        self.cycle_label = None
        self.gait_stats_text = None
        self.notebook = None
        self.filter_stats_label = None
        
        # Store plot references
        self.active_plots = {}
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI"""
        # Top menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open CSV", command=self.load_csv)
        file_menu.add_command(label="Open Video", command=self.load_video)
        file_menu.add_command(label="Load Reference Trajectory (.txt)", command=self.load_reference_trajectory)
        file_menu.add_separator()
        file_menu.add_command(label="Export Report", command=self.export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls & Gait Validator
        left_panel = tk.Frame(main_frame, bg='#1e1e1e', width=350, relief=tk.RIDGE, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        self.setup_left_panel(left_panel)
        
        # Middle panel - Video & 3D Plot
        middle_panel = tk.Frame(main_frame, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 5))
        
        self.setup_middle_panel(middle_panel)
        
        # Right panel - Additional Plots
        right_panel = tk.Frame(main_frame, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.setup_right_panel(right_panel)
        
    def setup_left_panel(self, panel):
        """Setup left control panel with gait validator"""
        # File info
        file_frame = tk.LabelFrame(panel, text="File Info", bg='#1e1e1e', 
                                   fg='white', font=('Arial', 10, 'bold'))
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_label = tk.Label(file_frame, text="No file loaded", 
                                   bg='#1e1e1e', fg='#888', 
                                   font=('Arial', 8), wraplength=330, justify=tk.LEFT)
        self.file_label.pack(padx=5, pady=5)
        
        tk.Button(file_frame, text="Load CSV File", command=self.load_csv,
                 bg='#4CAF50', fg='white', font=('Arial', 9, 'bold'),
                 width=25).pack(pady=2)
        
        tk.Button(file_frame, text="Load Video File", command=self.load_video,
                 bg='#2196F3', fg='white', font=('Arial', 9, 'bold'),
                 width=25).pack(pady=2)
        
        tk.Button(file_frame, text="Load Reference Trajectory", command=self.load_reference_trajectory,
                 bg='#FF9800', fg='white', font=('Arial', 9, 'bold'),
                 width=25).pack(pady=2)
        
        # Di setup_left_panel, setelah button Load Reference Trajectory
        tk.Button(file_frame, text="🎯 Filter Data", 
                command=lambda: self.nudge_data_to_reference(blend_strength=0.3),
                bg='#E91E63', fg='white', font=('Arial', 9, 'bold'),
                width=25).pack(pady=2)

        # Optional: Advanced nudge dengan slider
        def show_nudge_dialog():
            dialog = tk.Toplevel(self.root)
            dialog.title("Filter Settings")
            dialog.geometry("300x150")
            dialog.configure(bg='#1e1e1e')
            
            tk.Label(dialog, text="Filter Strength:", bg='#1e1e1e', 
                    fg='white', font=('Arial', 10)).pack(pady=10)
            
            strength_var = tk.DoubleVar(value=0.90)
            slider = tk.Scale(dialog, from_=0.0, to=1.0, resolution=0.05,
                            variable=strength_var, orient=tk.HORIZONTAL,
                            bg='#2b2b2b', fg='white', length=250)
            slider.pack(pady=5)
            
            tk.Label(dialog, text="0% - 100%", 
                    bg='#1e1e1e', fg='#888', font=('Arial', 8)).pack()
            
            tk.Button(dialog, text="Apply", 
                    command=lambda: [self.nudge_data_to_reference(strength_var.get()), 
                                    dialog.destroy()],
                    bg='#4CAF50', fg='white', font=('Arial', 9, 'bold'),
                    width=15).pack(pady=10)

        tk.Button(file_frame, text="⚙️ Advanced Filter", 
                command=show_nudge_dialog,
                bg='#9C27B0', fg='white', font=('Arial', 8),
                width=25).pack(pady=2)
        # Gait Cycle Validator
        gait_frame = tk.LabelFrame(panel, text="Gait Cycle Validator", bg='#1e1e1e',
                                   fg='white', font=('Arial', 10, 'bold'))
        gait_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Gait validation canvas
        self.gait_fig = plt.Figure(figsize=(4, 3.5), facecolor='#1e1e1e')
        self.gait_canvas = FigureCanvasTkAgg(self.gait_fig, master=gait_frame)
        self.gait_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Gait cycle controls
        cycle_control_frame = tk.Frame(gait_frame, bg='#1e1e1e')
        cycle_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(cycle_control_frame, text="◄", command=self.prev_gait_cycle,
                 bg='#555', fg='white', font=('Arial', 10, 'bold'),
                 width=3).pack(side=tk.LEFT, padx=2)
        
        self.cycle_label = tk.Label(cycle_control_frame, text="Cycle: 0/0",
                                   bg='#2b2b2b', fg='white', font=('Arial', 9),
                                   width=15)
        self.cycle_label.pack(side=tk.LEFT, padx=5)
        
        tk.Button(cycle_control_frame, text="►", command=self.next_gait_cycle,
                 bg='#555', fg='white', font=('Arial', 10, 'bold'),
                 width=3).pack(side=tk.LEFT, padx=2)
        
        # Gait statistics
        self.gait_stats_text = tk.Text(gait_frame, height=8, bg='#2b2b2b',
                                      fg='#00ff00', font=('Courier', 8),
                                      state=tk.DISABLED, wrap=tk.WORD)
        self.gait_stats_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Plot selection
        plot_frame = tk.LabelFrame(panel, text="Quick Plots", bg='#1e1e1e',
                                  fg='white', font=('Arial', 10, 'bold'))
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas and scrollbar
        plot_canvas = tk.Canvas(plot_frame, bg='#1e1e1e', highlightthickness=0)
        plot_scrollbar = tk.Scrollbar(plot_frame, orient="vertical", command=plot_canvas.yview)
        plot_scrollable_frame = tk.Frame(plot_canvas, bg='#1e1e1e')
        
        plot_scrollable_frame.bind(
            "<Configure>",
            lambda e: plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))
        )
        
        plot_canvas.create_window((0, 0), window=plot_scrollable_frame, anchor="nw")
        plot_canvas.configure(yscrollcommand=plot_scrollbar.set)
        
        # Pack scrollbar and canvas
        plot_scrollbar.pack(side="right", fill="y")
        plot_canvas.pack(side="left", fill="both", expand=True)
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            plot_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        plot_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        plots = [
            ("Position 3D", self.plot_3d_trajectory_full),
            ("Position vs Time", self.plot_position_vs_time),
            ("Ankle Velocity", self.plot_velocity),
            ("Angles Over Time", self.plot_angles),
            ("Speed Analysis", self.plot_speed),
            ("Angle Distribution", self.plot_angle_distribution),
            ("Correlation Matrix", self.plot_correlation),
            ("Trajectory Comparison", self.plot_trajectory_comparison),
            ("Deviation Analysis", self.plot_deviation_analysis),
            ("All Plots", self.plot_all),
        ]
        
        for text, cmd in plots:
            tk.Button(plot_scrollable_frame, text=text, command=cmd,
                     bg='#2196F3', fg='white', font=('Arial', 8),
                     width=28).pack(pady=2, padx=5)
            
    def setup_middle_panel(self, panel):
        """Setup middle panel with video and 3D plot"""
        # Video frame
        video_frame = tk.LabelFrame(panel, text="Video Playback", bg='#1e1e1e',
                                   fg='white', font=('Arial', 10, 'bold'))
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, bg='#000000', width=640, height=360)
        self.video_canvas.pack(padx=5, pady=5)
        
        # Video controls
        video_control_frame = tk.Frame(video_frame, bg='#1e1e1e')
        video_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.play_button = tk.Button(video_control_frame, text="▶ Play", 
                                     command=self.toggle_play,
                                     bg='#4CAF50', fg='white', 
                                     font=('Arial', 10, 'bold'), width=10)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        tk.Button(video_control_frame, text="◄◄", command=lambda: self.skip_frames(-30),
                 bg='#555', fg='white', font=('Arial', 9), width=4).pack(side=tk.LEFT, padx=2)
        
        tk.Button(video_control_frame, text="►►", command=lambda: self.skip_frames(30),
                 bg='#555', fg='white', font=('Arial', 9), width=4).pack(side=tk.LEFT, padx=2)
        
        self.frame_label = tk.Label(video_control_frame, text="Frame: 0/0",
                                   bg='#2b2b2b', fg='white', font=('Arial', 9))
        self.frame_label.pack(side=tk.LEFT, padx=10)
        
        # Frame slider - INITIALIZE HERE
        self.frame_slider = tk.Scale(video_control_frame, from_=0, to=100,
                                    orient=tk.HORIZONTAL, bg='#2b2b2b', fg='white',
                                    highlightthickness=0, command=self.on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 3D Trajectory plot
        traj_frame = tk.LabelFrame(panel, text="3D Trajectory", bg='#1e1e1e',
                                  fg='white', font=('Arial', 10, 'bold'))
        traj_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.traj_fig = plt.Figure(figsize=(7, 5), facecolor='#1e1e1e')
        self.traj_canvas = FigureCanvasTkAgg(self.traj_fig, master=traj_frame)
        self.traj_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def setup_right_panel(self, panel):
        """Setup right panel with additional plots"""
        # Notebook for multiple tabs
        self.notebook = ttk.Notebook(panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Style for dark theme
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background='#1e1e1e', borderwidth=0)
        style.configure('TNotebook.Tab', background='#2b2b2b', foreground='white',
                       padding=[10, 5])
        style.map('TNotebook.Tab', background=[('selected', '#4CAF50')])
    
    def load_csv(self):
        """Load CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.df = pd.read_csv(filename)
            self.current_file = filename
            
            # Detect file type
            self.is_dual_camera = False
            if 'ankle_x_3d' in self.df.columns:
                self.is_dual_camera = True
                rename_map = {
                    'ankle_x_3d': 'ankle_x', 'ankle_y_3d': 'ankle_y', 'ankle_z_3d': 'ankle_z',
                    'heel_x_3d': 'heel_x', 'heel_y_3d': 'heel_y', 'heel_z_3d': 'heel_z',
                    'toe_x_3d': 'toe_x', 'toe_y_3d': 'toe_y', 'toe_z_3d': 'toe_z'
                }
                for old_col, new_col in rename_map.items():
                    if old_col in self.df.columns:
                        self.df[new_col] = self.df[old_col]
                                 #===== AUTO FILTERING =====
            print("🔧 Applying data filtering...")
            
            # Calculate FPS
            fps = len(self.df) / self.df['timestamp'].max() if self.df['timestamp'].max() > 0 else 30
            
            # Filter ankle position
            self.df['ankle_x'] = DataFilter.clean_motion_data(
                self.df['ankle_x'].values, fps=fps, 
                outlier_sigma=3, kalman_R=2, kalman_Q=0.05, cutoff_hz=3
            )
            self.df['ankle_y'] = DataFilter.clean_motion_data(
                self.df['ankle_y'].values, fps=fps,
                outlier_sigma=3, kalman_R=1, kalman_Q=0.02, cutoff_hz=2  # Y lebih sensitive
            )
            self.df['ankle_z'] = DataFilter.clean_motion_data(
                self.df['ankle_z'].values, fps=fps,
                outlier_sigma=3, kalman_R=2, kalman_Q=0.05, cutoff_hz=3
            )
            
            # Filter heel and toe if available
            if 'heel_x' in self.df.columns:
                for point in ['heel', 'toe']:
                    self.df[f'{point}_x'] = DataFilter.clean_motion_data(self.df[f'{point}_x'], fps)
                    self.df[f'{point}_y'] = DataFilter.clean_motion_data(self.df[f'{point}_y'], fps, cutoff_hz=2)
                    self.df[f'{point}_z'] = DataFilter.clean_motion_data(self.df[f'{point}_z'], fps)
            
            print("✅ Filtering complete!")
            
            # Validate columns
            required_cols = ['timestamp', 'ankle_x', 'ankle_y', 'ankle_z',
                           'plantar_dorsi_angle', 'inversion_eversion_angle']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            if missing_cols:
                messagebox.showerror("Error", 
                    f"Missing required columns: {', '.join(missing_cols)}")
                return
            
            # Calculate speed if not present
            if 'ankle_speed' not in self.df.columns:
                self.calculate_speed()

            # Flip X axis (convert + to - and - to +)
            self.df['ankle_x'] = -self.df['ankle_x']
            
            # If heel and toe data exist, flip them too
            if 'heel_x' in self.df.columns:
                self.df['heel_x'] = -self.df['heel_x']
            
            if 'toe_x' in self.df.columns:
                self.df['toe_x'] = -self.df['toe_x']
            
            # Recalculate velocity after flipping X
            if 'ankle_velocity_x' in self.df.columns:
                self.df['ankle_velocity_x'] = -self.df['ankle_velocity_x']
                self.df['ankle_speed'] = np.sqrt(
                    self.df['ankle_velocity_x']**2 + 
                    self.df['ankle_velocity_y']**2 + 
                    self.df['ankle_velocity_z']**2
                )

            self.df['timestamp'] = self.df['timestamp'] - self.df['timestamp'].min()

            # Update file label
            file_type = "Dual Camera" if self.is_dual_camera else "Single Camera"
            self.file_label.config(
                text=f"CSV: {filename.split('/')[-1]}\nType: {file_type}\n"
                     f"Frames: {len(self.df)}\nDuration: {self.df['timestamp'].max():.2f}s",
                fg='white')
            
            Y_SCALE_FACTOR = 1.0  # Coba 2.0, 5.0, 10.0

            self.df['ankle_y'] = self.df['ankle_y'] * Y_SCALE_FACTOR

            if 'heel_y' in self.df.columns:
                self.df['heel_y'] = self.df['heel_y'] * Y_SCALE_FACTOR

            if 'toe_y' in self.df.columns:
                self.df['toe_y'] = self.df['toe_y'] * Y_SCALE_FACTOR

            # Recalculate velocity
            if 'ankle_velocity_y' in self.df.columns:
                self.df['ankle_velocity_y'] = self.df['ankle_velocity_y'] * Y_SCALE_FACTOR
                self.df['ankle_speed'] = np.sqrt(
                    self.df['ankle_velocity_x']**2 + 
                    self.df['ankle_velocity_y']**2 + 
                    self.df['ankle_velocity_z']**2
                )
            
            # Detect gait cycles
            self.detect_gait_cycles()
            
            # Update displays
            self.update_3d_trajectory()
            self.update_gait_validator()
            
            # Setup frame slider
            if self.video_cap:
                self.frame_slider.config(to=min(len(self.df)-1, int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))-1))
            else:
                self.frame_slider.config(to=len(self.df)-1)
            
            messagebox.showinfo("Success", "CSV file loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")
    
    def load_video(self):
        """Load video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            if self.video_cap:
                self.video_cap.release()
            
            self.video_cap = cv2.VideoCapture(filename)
            self.video_file = filename
            
            if not self.video_cap.isOpened():
                raise Exception("Cannot open video file")
            
            # Get video properties
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            
            # Update file label
            current_text = self.file_label.cget("text")
            self.file_label.config(
                text=f"{current_text}\n\nVideo: {filename.split('/')[-1]}\n"
                     f"Frames: {total_frames}\nFPS: {self.video_fps:.1f}")
            
            # Setup slider
            if self.df is not None:
                self.frame_slider.config(to=min(len(self.df)-1, total_frames-1))
            else:
                self.frame_slider.config(to=total_frames-1)
            
            # Show first frame
            self.current_frame_idx = 0
            self.update_video_frame()
            
            messagebox.showinfo("Success", "Video loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video:\n{str(e)}")
    
    def normalize_trajectory(self, x, y):
        """Normalize trajectory to 0-1 range based on shape, not absolute size"""
        # Center the trajectory
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        
        # Normalize to unit scale (preserving shape)
        x_range = np.max(x_centered) - np.min(x_centered)
        y_range = np.max(y_centered) - np.min(y_centered)
        
        # Use the larger range to preserve aspect ratio
        max_range = max(x_range, y_range)
        
        if max_range > 0:
            x_norm = x_centered / max_range
            y_norm = y_centered / max_range
        else:
            x_norm = x_centered
            y_norm = y_centered
        
        return x_norm, y_norm
    
    def align_trajectories(self, actual_x, actual_y, ref_x, ref_y):
        """
        Align both trajectories to start from (0, 0) for pure shape comparison
        Returns aligned trajectories with the same scale
        """
        # Translate actual to start from (0, 0)
        actual_x_aligned = actual_x - actual_x.iloc[0] if hasattr(actual_x, 'iloc') else actual_x - actual_x[0]
        actual_y_aligned = actual_y - actual_y.iloc[0] if hasattr(actual_y, 'iloc') else actual_y - actual_y[0]
        
        # Translate reference to start from (0, 0)
        ref_x_aligned = ref_x - ref_x.iloc[0] if hasattr(ref_x, 'iloc') else ref_x - ref_x[0]
        ref_y_aligned = ref_y - ref_y.iloc[0] if hasattr(ref_y, 'iloc') else ref_y - ref_y[0]
        
        return actual_x_aligned, actual_y_aligned, ref_x_aligned, ref_y_aligned
    
    def calculate_rmse(self, actual_x, actual_y, ref_x, ref_y, aligned=True):
        """
        Calculate RMSE between actual and reference trajectories
        
        Parameters:
        - aligned: If True, align trajectories first before calculating RMSE
        
        Returns:
        - rmse_total: Overall RMSE (mm)
        - rmse_x: X-component RMSE (mm)
        - rmse_y: Y-component RMSE (mm)
        """
        if aligned:
            # Align trajectories first for shape comparison
            actual_x_aligned, actual_y_aligned, ref_x_aligned, ref_y_aligned = \
                self.align_trajectories(actual_x, actual_y, ref_x, ref_y)
        else:
            # Use raw trajectories (absolute position comparison)
            actual_x_aligned = actual_x
            actual_y_aligned = actual_y
            ref_x_aligned = ref_x
            ref_y_aligned = ref_y
        
        # Interpolate actual data to match reference length
        actual_indices = np.linspace(0, len(actual_x_aligned)-1, len(ref_x_aligned))
        actual_x_interp = np.interp(actual_indices, np.arange(len(actual_x_aligned)), actual_x_aligned)
        actual_y_interp = np.interp(actual_indices, np.arange(len(actual_y_aligned)), actual_y_aligned)
        
        # Calculate squared errors
        x_errors = (actual_x_interp - ref_x_aligned.values) ** 2
        y_errors = (actual_y_interp - ref_y_aligned.values) ** 2
        
        # RMSE calculations
        rmse_x = np.sqrt(np.mean(x_errors)) * 1000  # Convert to mm
        rmse_y = np.sqrt(np.mean(y_errors)) * 1000  # Convert to mm
        rmse_total = np.sqrt(np.mean(x_errors + y_errors)) * 1000  # Total RMSE in mm
        
        return rmse_total, rmse_x, rmse_y
    
    def load_reference_trajectory(self):
        """Load reference trajectory from .txt file"""
        filename = filedialog.askopenfilename(
            title="Select Reference Trajectory File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Read the txt file
            ref_data = pd.read_csv(filename, header=0, names=['x', 'y'])
            
            # Convert from mm to ms
            ref_data['x'] = ref_data['x'] / 50.0
            ref_data['y'] = ref_data['y'] / 50.0
            
            self.reference_trajectory = ref_data
            self.reference_file = filename
            
            # Update file label
            current_text = self.file_label.cget("text")
            self.file_label.config(
                text=f"{current_text}\n\nReference: {filename.split('/')[-1]}\n"
                     f"Points: {len(ref_data)}")
            
            # Update displays if data is loaded
            if self.df is not None:
                self.update_gait_validator()
                self.update_3d_trajectory()
            
            messagebox.showinfo("Success", 
                f"Reference trajectory loaded!\n"
                f"Points: {len(ref_data)}\n"
                f"X range: [{ref_data['x'].min():.3f}, {ref_data['x'].max():.3f}] m\n"
                f"Y range: [{ref_data['y'].min():.3f}, {ref_data['y'].max():.3f}] m")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load reference trajectory:\n{str(e)}")
    
    
    def calculate_speed(self):
        """Calculate speed from position data"""
        dt = np.diff(self.df['timestamp'], prepend=self.df['timestamp'].iloc[0])
        dt[0] = dt[1]
        
        vx = np.gradient(self.df['ankle_x'], self.df['timestamp'])
        vy = np.gradient(self.df['ankle_y'], self.df['timestamp'])
        vz = np.gradient(self.df['ankle_z'], self.df['timestamp'])
        
        self.df['ankle_velocity_x'] = vx
        self.df['ankle_velocity_y'] = vy
        self.df['ankle_velocity_z'] = vz
        self.df['ankle_speed'] = np.sqrt(vx**2 + vy**2 + vz**2)
    
    def detect_gait_cycles(self):
        """Detect gait cycles from ankle height (Y position)"""
        if self.df is None:
            return
        
        self.gait_cycles = []
        ankle_y = self.df['ankle_y'].values
        
        peaks, _ = find_peaks(-ankle_y, distance=10, prominence=0.01)
        
        for i in range(len(peaks)-1):
            start_idx = peaks[i]
            end_idx = peaks[i+1]
            cycle_data = self.df.iloc[start_idx:end_idx+1]
            
            cycle = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration': cycle_data['timestamp'].iloc[-1] - cycle_data['timestamp'].iloc[0],
                'stride_length': np.sqrt(
                    (cycle_data['ankle_x'].iloc[-1] - cycle_data['ankle_x'].iloc[0])**2 +
                    (cycle_data['ankle_z'].iloc[-1] - cycle_data['ankle_z'].iloc[0])**2
                ),
                'max_height': cycle_data['ankle_y'].max(),
                'pd_rom': cycle_data['plantar_dorsi_angle'].max() - cycle_data['plantar_dorsi_angle'].min(),
                'ie_rom': cycle_data['inversion_eversion_angle'].max() - cycle_data['inversion_eversion_angle'].min(),
            }
            self.gait_cycles.append(cycle)
        
        if self.gait_cycles:
            self.current_cycle_idx = 0
            self.cycle_label.config(text=f"Cycle: 1/{len(self.gait_cycles)}")
    
    def update_gait_validator(self):
        """Update gait cycle validator plot"""
        if not self.gait_cycles or self.df is None:
            return
        
        self.gait_fig.clear()
        ax = self.gait_fig.add_subplot(111, facecolor='#2b2b2b')
        
        for i, cycle in enumerate(self.gait_cycles):
            cycle_data = self.df.iloc[cycle['start_idx']:cycle['end_idx']+1]
            time_norm = np.linspace(0, 100, len(cycle_data))
            
            if i == self.current_cycle_idx:
                ax.plot(time_norm, cycle_data['ankle_y'], 'lime', linewidth=3, 
                       label=f"Cycle {i+1} (Current)", alpha=0.9)
            else:
                ax.plot(time_norm, cycle_data['ankle_y'], 'gray', linewidth=1, alpha=0.3)
        
        all_cycles_y = []
        for cycle in self.gait_cycles:
            cycle_data = self.df.iloc[cycle['start_idx']:cycle['end_idx']+1]
            all_cycles_y.append(np.interp(np.linspace(0, 100, 100), 
                                         np.linspace(0, 100, len(cycle_data)),
                                         cycle_data['ankle_y']))
        
        mean_cycle = np.mean(all_cycles_y, axis=0)
        ax.plot(np.linspace(0, 100, 100), mean_cycle, 'r--', linewidth=2, 
               label='Mean Cycle', alpha=0.8)
        
        # Plot reference trajectory if loaded (NORMALIZED to match scale)
        if self.reference_trajectory is not None:
            ref_x_norm = np.linspace(0, 100, len(self.reference_trajectory))
            # Normalize reference Y to match the scale of mean_cycle
            ref_y = self.reference_trajectory['y'].values
            ref_y_norm = (ref_y - ref_y.min()) / (ref_y.max() - ref_y.min())  # 0-1
            ref_y_scaled = ref_y_norm * (mean_cycle.max() - mean_cycle.min()) + mean_cycle.min()  # Scale to match
            
            ax.plot(ref_x_norm, ref_y_scaled, 'gold', linewidth=3, 
                   label='Reference (Shape-Matched)', alpha=0.9, linestyle='-.')
        
        ax.set_xlabel('Gait Cycle (%)', color='white', fontsize=9)
        ax.set_ylabel('Ankle Height (m)', color='white', fontsize=9)
        ax.set_title('Gait Cycle Validation', color='white', fontsize=10, fontweight='bold')
        ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, alpha=0.3)
        
        self.gait_canvas.draw()
        self.update_gait_statistics()
    
    def update_gait_statistics(self):
        """Update gait cycle statistics text"""
        if not self.gait_cycles:
            return
        
        cycle = self.gait_cycles[self.current_cycle_idx]
        avg_duration = np.mean([c['duration'] for c in self.gait_cycles])
        avg_stride = np.mean([c['stride_length'] for c in self.gait_cycles])
        similarity = 100 * (1 - abs(cycle['duration'] - avg_duration) / avg_duration)
        
        stats = f"""
CURRENT CYCLE STATS

Duration: {cycle['duration']:.3f} s
Stride Length: {cycle['stride_length']:.4f} m
Max Height: {cycle['max_height']:.4f} m

P/D ROM: {cycle['pd_rom']:.2f}°
I/E ROM: {cycle['ie_rom']:.2f}°

Similarity: {similarity:.1f}%

AVG ACROSS ALL CYCLES
Avg Duration: {avg_duration:.3f} s
Avg Stride: {avg_stride:.4f} m
Total Cycles: {len(self.gait_cycles)}
        """
        
        self.gait_stats_text.config(state=tk.NORMAL)
        self.gait_stats_text.delete('1.0', tk.END)
        self.gait_stats_text.insert('1.0', stats)
        self.gait_stats_text.config(state=tk.DISABLED)
    
    def prev_gait_cycle(self):
        """Go to previous gait cycle"""
        if not self.gait_cycles:
            return
        
        self.current_cycle_idx = (self.current_cycle_idx - 1) % len(self.gait_cycles)
        self.cycle_label.config(text=f"Cycle: {self.current_cycle_idx+1}/{len(self.gait_cycles)}")
        
        cycle = self.gait_cycles[self.current_cycle_idx]
        self.current_frame_idx = cycle['start_idx']
        self.frame_slider.set(self.current_frame_idx)
        
        self.update_gait_validator()
        self.update_video_frame()
        self.update_3d_trajectory()
    
    def next_gait_cycle(self):
        """Go to next gait cycle"""
        if not self.gait_cycles:
            return
        
        self.current_cycle_idx = (self.current_cycle_idx + 1) % len(self.gait_cycles)
        self.cycle_label.config(text=f"Cycle: {self.current_cycle_idx+1}/{len(self.gait_cycles)}")
        
        cycle = self.gait_cycles[self.current_cycle_idx]
        self.current_frame_idx = cycle['start_idx']
        self.frame_slider.set(self.current_frame_idx)
        
        self.update_gait_validator()
        self.update_video_frame()
        self.update_3d_trajectory()
    
    def update_3d_trajectory(self):
        """Update 3D trajectory plot with current position marker"""
        if self.df is None:
            return
        
        self.traj_fig.clear()
        ax = self.traj_fig.add_subplot(111, projection='3d', facecolor='#2b2b2b')
        
        required_cols = ['ankle_x', 'ankle_y', 'ankle_z']
        if not all(col in self.df.columns for col in required_cols):
            return
        
        ax.plot(self.df['ankle_x'], self.df['ankle_y'], self.df['ankle_z'],
               'r-', linewidth=1, label='Ankle', alpha=0.4)
        
        if all(col in self.df.columns for col in ['heel_x', 'heel_y', 'heel_z']):
            ax.plot(self.df['heel_x'], self.df['heel_y'], self.df['heel_z'],
                   'g-', linewidth=1, label='Heel', alpha=0.3)
        
        if all(col in self.df.columns for col in ['toe_x', 'toe_y', 'toe_z']):
            ax.plot(self.df['toe_x'], self.df['toe_y'], self.df['toe_z'],
                   'b-', linewidth=1, label='Toe', alpha=0.3)
        
        if self.gait_cycles and self.current_cycle_idx < len(self.gait_cycles):
            cycle = self.gait_cycles[self.current_cycle_idx]
            cycle_data = self.df.iloc[cycle['start_idx']:cycle['end_idx']+1]
            
            ax.plot(cycle_data['ankle_x'], cycle_data['ankle_y'], cycle_data['ankle_z'],
                   'yellow', linewidth=3, label='Current Cycle', alpha=0.9)
        
        # Plot reference trajectory if loaded (as 2D on ground plane)
        if self.reference_trajectory is not None:
            # Assume z=0 for reference (ground plane)
            z_ref = np.zeros(len(self.reference_trajectory))
            ax.plot(self.reference_trajectory['x'], self.reference_trajectory['y'], z_ref,
                   'gold', linewidth=3, label='Reference (Ground)', alpha=0.7, linestyle='-.')
        
        if 0 <= self.current_frame_idx < len(self.df):
            current = self.df.iloc[self.current_frame_idx]
            ax.scatter(current['ankle_x'], current['ankle_y'], current['ankle_z'],
                      c='cyan', s=150, marker='o', label='Current Position',
                      edgecolors='white', linewidth=2)
        
        ax.scatter(self.df['ankle_x'].iloc[0], self.df['ankle_y'].iloc[0], 
                  self.df['ankle_z'].iloc[0], c='lime', s=100, marker='o', 
                  label='Start', edgecolors='black', linewidth=2)
        ax.scatter(self.df['ankle_x'].iloc[-1], self.df['ankle_y'].iloc[-1], 
                  self.df['ankle_z'].iloc[-1], c='orange', s=100, marker='s', 
                  label='End', edgecolors='black', linewidth=2)
        
        ax.set_xlabel('X (m)', color='white', fontsize=9)
        ax.set_ylabel('Y (m)', color='white', fontsize=9)
        ax.set_zlabel('Z (m)', color='white', fontsize=9)
        ax.set_title('3D Foot Trajectory (Synced)', color='white', fontsize=11, fontweight='bold')
        ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8)
        
        ax.tick_params(colors='white', labelsize=8)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)
        
        self.traj_canvas.draw()
    
    def update_video_frame(self):
        """Update video frame display"""
        if not self.video_cap:
            return
        
        try:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.video_cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.df is not None and self.current_frame_idx < len(self.df):
                    current = self.df.iloc[self.current_frame_idx]
                    
                    cv2.putText(frame, f"Frame: {self.current_frame_idx}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Time: {current['timestamp']:.2f}s", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"P/D Angle: {current['plantar_dorsi_angle']:.1f}°", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    if 'ankle_speed' in self.df.columns:
                        cv2.putText(frame, f"Speed: {current['ankle_speed']:.3f} m/s", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    h, w = frame.shape[:2]
                    scale = min(canvas_width/w, canvas_height/h)
                    new_w = int(w * scale * 0.95)
                    new_h = int(h * scale * 0.95)
                    frame = cv2.resize(frame, (new_w, new_h))
                
                img = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=img)
                
                self.video_canvas.delete("all")
                self.video_canvas.create_image(
                    canvas_width//2, canvas_height//2, 
                    image=photo, anchor=tk.CENTER
                )
                self.video_canvas.image = photo
            
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_label.config(text=f"Frame: {self.current_frame_idx}/{total_frames}")
            
        except Exception as e:
            print(f"Error updating video frame: {e}")
    
    def toggle_play(self):
        """Toggle video playback"""
        if not self.video_cap or self.df is None:
            messagebox.showwarning("Warning", "Please load both CSV and video files!")
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_button.config(text="⏸ Pause", bg='#FF9800')
            self.play_video()
        else:
            self.play_button.config(text="▶ Play", bg='#4CAF50')
    
    def play_video(self):
        """Play video with synchronized data"""
        if not self.is_playing:
            return
        
        total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.current_frame_idx < min(len(self.df)-1, total_frames-1):
            self.current_frame_idx += 1
            self.frame_slider.set(self.current_frame_idx)
            self.update_video_frame()
            self.update_3d_trajectory()
            self.update_all_open_plots()  # Update semua plot yang terbuka
            
            delay = int(1000 / self.video_fps)
            self.root.after(delay, self.play_video)
        else:
            self.is_playing = False
            self.play_button.config(text="▶ Play", bg='#4CAF50')
    
    def skip_frames(self, n):
        """Skip forward or backward by n frames"""
        if not self.video_cap or self.df is None:
            return
        
        total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frame = min(len(self.df)-1, total_frames-1)
        
        self.current_frame_idx = max(0, min(self.current_frame_idx + n, max_frame))
        self.frame_slider.set(self.current_frame_idx)
        self.update_video_frame()
        self.update_3d_trajectory()
        self.update_all_open_plots()  # Update semua plot saat skip
    
    def on_slider_change(self, value):
        """Handle slider value change"""
        self.current_frame_idx = int(float(value))
        self.update_video_frame()
        self.update_3d_trajectory()
        self.update_all_open_plots()  # Update semua plot saat slider berubah
    
    def create_plot_tab(self, title):
        """Create a new tab with close button"""
        tab = tk.Frame(self.notebook, bg='#1e1e1e')
        
        close_frame = tk.Frame(tab, bg='#1e1e1e')
        close_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        close_btn = tk.Button(close_frame, text="✕ Close Tab", 
                             command=lambda: self.close_plot_tab(title),
                             bg='#d32f2f', fg='white', font=('Arial', 8, 'bold'),
                             width=12, cursor='hand2')
        close_btn.pack(side=tk.RIGHT)
        
        tk.Label(close_frame, text=title, bg='#1e1e1e', fg='white',
                font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        
        self.notebook.add(tab, text=title)
        
        fig = plt.Figure(figsize=(10, 7), facecolor='#1e1e1e')
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.active_plots[title] = {'fig': fig, 'canvas': canvas, 'axes': [], 'tab': tab}
        
        self.notebook.select(tab)
        
        return fig
    
    def close_plot_tab(self, title):
        """Close a specific plot tab"""
        if title not in self.active_plots:
            return
        
        try:
            tab_widget = self.active_plots[title]['tab']
            
            for i, tab_id in enumerate(self.notebook.tabs()):
                if self.notebook.nametowidget(tab_id) == tab_widget:
                    self.notebook.forget(i)
                    break
            
            self.active_plots[title]['fig'].clear()
            plt.close(self.active_plots[title]['fig'])
            
            del self.active_plots[title]
            
        except Exception as e:
            print(f"Error closing tab '{title}': {e}")
    
    def close_all_tabs(self):
        """Close all open plot tabs"""
        tabs_to_close = list(self.active_plots.keys())
        for title in tabs_to_close:
            self.close_plot_tab(title)
        messagebox.showinfo("Success", "All plot tabs closed!")
    
    def update_all_open_plots(self):
        """Update all currently open plot tabs"""
        if self.df is None:
            return
        
        for plot_name, plot_data in self.active_plots.items():
            if plot_name == "Angle Analysis":
                self.update_angle_plot_realtime(plot_data)
            elif plot_name == "Speed Analysis":
                self.update_speed_plot_realtime(plot_data)
            elif plot_name == "Position vs Time":
                self.update_position_plot_realtime(plot_data)
            elif plot_name == "Ankle Velocity":
                self.update_velocity_plot_realtime(plot_data)
    
    def plot_3d_trajectory_full(self):
        """Plot full 3D trajectory in new tab"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first!")
            return
        
        fig = self.create_plot_tab("3D Trajectory Full")
        ax = fig.add_subplot(111, projection='3d', facecolor='#2b2b2b')
        
        ax.plot(self.df['ankle_x'], self.df['ankle_y'], self.df['ankle_z'],
               'r-', linewidth=2, label='Ankle', alpha=0.7)
        
        if all(col in self.df.columns for col in ['heel_x', 'heel_y', 'heel_z']):
            ax.plot(self.df['heel_x'], self.df['heel_y'], self.df['heel_z'],
                   'g-', linewidth=2, label='Heel', alpha=0.6)
        
        if all(col in self.df.columns for col in ['toe_x', 'toe_y', 'toe_z']):
            ax.plot(self.df['toe_x'], self.df['toe_y'], self.df['toe_z'],
                   'b-', linewidth=2, label='Toe', alpha=0.6)
        
        ax.scatter(self.df['ankle_x'].iloc[0], self.df['ankle_y'].iloc[0], 
                  self.df['ankle_z'].iloc[0], c='lime', s=150, marker='o', 
                  label='Start', edgecolors='black', linewidth=2)
        ax.scatter(self.df['ankle_x'].iloc[-1], self.df['ankle_y'].iloc[-1], 
                  self.df['ankle_z'].iloc[-1], c='orange', s=150, marker='s', 
                  label='End', edgecolors='black', linewidth=2)
        
        ax.set_xlabel('X (m)', color='white', fontsize=11)
        ax.set_ylabel('Y (m)', color='white', fontsize=11)
        ax.set_zlabel('Z (m)', color='white', fontsize=11)
        ax.set_title('Full 3D Foot Trajectory', color='white', fontsize=14, fontweight='bold')
        ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=10)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
    
    def plot_position_vs_time(self):
        """Plot position vs time"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first!")
            return
        
        fig = self.create_plot_tab("Position vs Time")
        
        ax1 = fig.add_subplot(3, 1, 1, facecolor='#2b2b2b')
        ax1.plot(self.df['timestamp'], self.df['ankle_x'], 'r-', linewidth=2, label='Ankle X')
        ax1.set_ylabel('X Position (m)', color='white', fontsize=10)
        ax1.set_title('Ankle Position Components Over Time', color='white', fontsize=13, fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)
        ax1.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        
        ax2 = fig.add_subplot(3, 1, 2, facecolor='#2b2b2b')
        ax2.plot(self.df['timestamp'], self.df['ankle_y'], 'g-', linewidth=2, label='Ankle Y')
        ax2.set_ylabel('Y Position (m)', color='white', fontsize=10)
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        ax2.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        
        ax3 = fig.add_subplot(3, 1, 3, facecolor='#2b2b2b')
        ax3.plot(self.df['timestamp'], self.df['ankle_z'], 'b-', linewidth=2, label='Ankle Z')
        ax3.set_xlabel('Time (s)', color='white', fontsize=10)
        ax3.set_ylabel('Z Position (m)', color='white', fontsize=10)
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.3)
        ax3.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        
        # Store axes for updates
        self.active_plots["Position vs Time"]['axes'] = [ax1, ax2, ax3]
        
        fig.tight_layout()
        
        # Draw current position marker
        self.update_position_plot_realtime(self.active_plots["Position vs Time"])
    
    def update_position_plot_realtime(self, plot_data):
        """Update position plot with current frame marker"""
        if self.df is None or 0 > self.current_frame_idx or self.current_frame_idx >= len(self.df):
            return
        
        axes = plot_data['axes']
        if len(axes) != 3:
            return
        
        current = self.df.iloc[self.current_frame_idx]
        
        # Clear previous markers (lines with label starting with '_')
        for ax in axes:
            lines_to_remove = [line for line in ax.lines if len(line.get_label()) > 0 and line.get_label().startswith('_marker')]
            for line in lines_to_remove:
                line.remove()
            artists_to_remove = [artist for artist in ax.collections if hasattr(artist, '_label') and artist._label and artist._label.startswith('_marker')]
            for artist in artists_to_remove:
                artist.remove()
        
        # Add new markers
        axes[0].axvline(x=current['timestamp'], color='cyan', linestyle='--', linewidth=2, label='_marker_line')
        axes[0].scatter(current['timestamp'], current['ankle_x'], c='cyan', s=100, zorder=5, 
                       edgecolors='white', linewidth=2, label='_marker_point')
        
        axes[1].axvline(x=current['timestamp'], color='cyan', linestyle='--', linewidth=2, label='_marker_line')
        axes[1].scatter(current['timestamp'], current['ankle_y'], c='cyan', s=100, zorder=5, 
                       edgecolors='white', linewidth=2, label='_marker_point')
        
        axes[2].axvline(x=current['timestamp'], color='cyan', linestyle='--', linewidth=2, label='_marker_line')
        axes[2].scatter(current['timestamp'], current['ankle_z'], c='cyan', s=100, zorder=5, 
                       edgecolors='white', linewidth=2, label='_marker_point')
        
        plot_data['canvas'].draw_idle()
    
    def plot_velocity(self):
        """Plot velocity components"""
        if self.df is None or 'ankle_velocity_x' not in self.df.columns:
            messagebox.showwarning("Warning", "Velocity data not available!")
            return
        
        fig = self.create_plot_tab("Ankle Velocity")
        
        ax1 = fig.add_subplot(2, 2, 1, facecolor='#2b2b2b')
        ax1.plot(self.df['timestamp'], self.df['ankle_velocity_x'], 'r-', linewidth=1.5)
        ax1.set_ylabel('Vx (m/s)', color='white')
        ax1.set_title('X Velocity', color='white', fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(2, 2, 2, facecolor='#2b2b2b')
        ax2.plot(self.df['timestamp'], self.df['ankle_velocity_y'], 'g-', linewidth=1.5)
        ax2.set_ylabel('Vy (m/s)', color='white')
        ax2.set_title('Y Velocity', color='white', fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(2, 2, 3, facecolor='#2b2b2b')
        ax3.plot(self.df['timestamp'], self.df['ankle_velocity_z'], 'b-', linewidth=1.5)
        ax3.set_xlabel('Time (s)', color='white')
        ax3.set_ylabel('Vz (m/s)', color='white')
        ax3.set_title('Z Velocity', color='white', fontweight='bold')
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(2, 2, 4, facecolor='#2b2b2b')
        if 'ankle_speed' in self.df.columns:
            ax4.plot(self.df['timestamp'], self.df['ankle_speed'], 'yellow', linewidth=2)
        ax4.set_xlabel('Time (s)', color='white')
        ax4.set_ylabel('Speed (m/s)', color='white')
        ax4.set_title('Total Speed', color='white', fontweight='bold')
        ax4.tick_params(colors='white')
        ax4.grid(True, alpha=0.3)
        
        # Store axes for updates
        self.active_plots["Ankle Velocity"]['axes'] = [ax1, ax2, ax3, ax4]
        
        fig.tight_layout()
        
        # Draw current position marker
        self.update_velocity_plot_realtime(self.active_plots["Ankle Velocity"])
    
    def update_velocity_plot_realtime(self, plot_data):
        """Update velocity plot with current frame marker"""
        if self.df is None or 0 > self.current_frame_idx or self.current_frame_idx >= len(self.df):
            return
        
        axes = plot_data['axes']
        if len(axes) != 4:
            return
        
        current = self.df.iloc[self.current_frame_idx]
        
        # Clear previous markers
        for ax in axes:
            lines_to_remove = [line for line in ax.lines if len(line.get_label()) > 0 and line.get_label().startswith('_marker')]
            for line in lines_to_remove:
                line.remove()
            artists_to_remove = [artist for artist in ax.collections if hasattr(artist, '_label') and artist._label and artist._label.startswith('_marker')]
            for artist in artists_to_remove:
                artist.remove()
        
        # Add new markers
        axes[0].axvline(x=current['timestamp'], color='cyan', linestyle='--', linewidth=2, label='_marker_line')
        axes[0].scatter(current['timestamp'], current['ankle_velocity_x'], c='cyan', s=80, zorder=5, 
                       edgecolors='white', linewidth=2, label='_marker_point')
        
        axes[1].axvline(x=current['timestamp'], color='cyan', linestyle='--', linewidth=2, label='_marker_line')
        axes[1].scatter(current['timestamp'], current['ankle_velocity_y'], c='cyan', s=80, zorder=5, 
                       edgecolors='white', linewidth=2, label='_marker_point')
        
        axes[2].axvline(x=current['timestamp'], color='cyan', linestyle='--', linewidth=2, label='_marker_line')
        axes[2].scatter(current['timestamp'], current['ankle_velocity_z'], c='cyan', s=80, zorder=5, 
                       edgecolors='white', linewidth=2, label='_marker_point')
        
        if 'ankle_speed' in self.df.columns:
            axes[3].axvline(x=current['timestamp'], color='cyan', linestyle='--', linewidth=2, label='_marker_line')
            axes[3].scatter(current['timestamp'], current['ankle_speed'], c='cyan', s=80, zorder=5, 
                           edgecolors='white', linewidth=2, label='_marker_point')
        
        plot_data['canvas'].draw_idle()
    
    def plot_angles(self):
        """Plot angles over time"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first!")
            return
        
        fig = self.create_plot_tab("Angle Analysis")
        
        ax1 = fig.add_subplot(2, 1, 1, facecolor='#2b2b2b')
        ax1.plot(self.df['timestamp'], self.df['plantar_dorsi_angle'], 
                'lime', linewidth=2)
        ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax1.fill_between(self.df['timestamp'], self.df['plantar_dorsi_angle'], 
                        alpha=0.3, color='lime')
        ax1.set_xlabel('Time (s)', color='white', fontsize=11)
        ax1.set_ylabel('Angle (degrees)', color='white', fontsize=11)
        ax1.set_title('Plantar/Dorsiflexion Angle Over Time', 
                     color='white', fontsize=13, fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(2, 1, 2, facecolor='#2b2b2b')
        ax2.plot(self.df['timestamp'], self.df['inversion_eversion_angle'], 
                'deepskyblue', linewidth=2)
        ax2.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax2.fill_between(self.df['timestamp'], self.df['inversion_eversion_angle'], 
                        alpha=0.3, color='deepskyblue')
        ax2.set_xlabel('Time (s)', color='white', fontsize=11)
        ax2.set_ylabel('Angle (degrees)', color='white', fontsize=11)
        ax2.set_title('Inversion/Eversion Angle Over Time', 
                     color='white', fontsize=13, fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        
        # Store axes for updates
        self.active_plots["Angle Analysis"]['axes'] = [ax1, ax2]
        
        fig.tight_layout()
        
        # Draw current position marker
        self.update_angle_plot_realtime(self.active_plots["Angle Analysis"])
    
    def update_angle_plot_realtime(self, plot_data):
        """Update angle plot with current frame marker"""
        if self.df is None or 0 > self.current_frame_idx or self.current_frame_idx >= len(self.df):
            return
        
        axes = plot_data['axes']
        if len(axes) != 2:
            return
        
        current = self.df.iloc[self.current_frame_idx]
        
        # Clear previous markers
        for ax in axes:
            lines_to_remove = [line for line in ax.lines if len(line.get_label()) > 0 and line.get_label().startswith('_marker')]
            for line in lines_to_remove:
                line.remove()
            artists_to_remove = [artist for artist in ax.collections if hasattr(artist, '_label') and artist._label and artist._label.startswith('_marker')]
            for artist in artists_to_remove:
                artist.remove()
        
        # Add new markers
        axes[0].axvline(x=current['timestamp'], color='cyan', linestyle='--', linewidth=2, label='_marker_line')
        axes[0].scatter(current['timestamp'], current['plantar_dorsi_angle'],
                       c='cyan', s=100, zorder=5, edgecolors='white', linewidth=2, label='_marker_point')
        
        axes[1].axvline(x=current['timestamp'], color='cyan', linestyle='--', linewidth=2, label='_marker_line')
        axes[1].scatter(current['timestamp'], current['inversion_eversion_angle'],
                       c='cyan', s=100, zorder=5, edgecolors='white', linewidth=2, label='_marker_point')
        
        plot_data['canvas'].draw_idle()
    
    def plot_speed(self):
        """Plot speed analysis"""
        if self.df is None or 'ankle_speed' not in self.df.columns:
            messagebox.showwarning("Warning", "Speed data not available!")
            return
        
        fig = self.create_plot_tab("Speed Analysis")
        
        ax1 = fig.add_subplot(2, 2, 1, facecolor='#2b2b2b')
        ax1.plot(self.df['timestamp'], self.df['ankle_speed'], 
                'yellow', linewidth=2)
        ax1.set_xlabel('Time (s)', color='white')
        ax1.set_ylabel('Speed (m/s)', color='white')
        ax1.set_title('Ankle Speed Over Time', color='white', fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(2, 2, 2, facecolor='#2b2b2b')
        dt = np.diff(self.df['timestamp'], prepend=0)
        cumulative_dist = np.cumsum(self.df['ankle_speed'] * dt)
        ax2.plot(self.df['timestamp'], cumulative_dist, 'cyan', linewidth=2)
        ax2.fill_between(self.df['timestamp'], cumulative_dist, 
                        alpha=0.3, color='cyan')
        ax2.set_xlabel('Time (s)', color='white')
        ax2.set_ylabel('Distance (m)', color='white')
        ax2.set_title(f'Cumulative Distance: {cumulative_dist[-1]:.3f}m', 
                     color='white', fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(2, 2, 3, facecolor='#2b2b2b')
        ax3.hist(self.df['ankle_speed'], bins=30, color='yellow', 
                alpha=0.7, edgecolor='white')
        ax3.set_xlabel('Speed (m/s)', color='white')
        ax3.set_ylabel('Frequency', color='white')
        ax3.set_title('Speed Distribution', color='white', fontweight='bold')
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.3, axis='y')
        
        ax4 = fig.add_subplot(2, 2, 4, facecolor='#2b2b2b')
        scatter = ax4.scatter(self.df['plantar_dorsi_angle'], 
                            self.df['ankle_speed'],
                            c=self.df['timestamp'], cmap='viridis', 
                            alpha=0.6, s=20)
        ax4.set_xlabel('Plantar/Dorsi Angle (deg)', color='white')
        ax4.set_ylabel('Speed (m/s)', color='white')
        ax4.set_title('Speed vs Plantar/Dorsi Angle', color='white', fontweight='bold')
        ax4.tick_params(colors='white')
        ax4.grid(True, alpha=0.3)
        cbar = fig.colorbar(scatter, ax=ax4)
        cbar.set_label('Time (s)', color='white')
        cbar.ax.tick_params(colors='white')
        
        # Store axes and cumulative distance data for updates
        self.active_plots["Speed Analysis"]['axes'] = [ax1, ax2, ax3, ax4]
        self.active_plots["Speed Analysis"]['cumulative_dist'] = cumulative_dist
        
        fig.tight_layout()
        
        # Draw current position marker
        self.update_speed_plot_realtime(self.active_plots["Speed Analysis"])
    
    def update_speed_plot_realtime(self, plot_data):
        """Update speed plot with current frame marker"""
        if self.df is None or 0 > self.current_frame_idx or self.current_frame_idx >= len(self.df):
            return
        
        axes = plot_data['axes']
        if len(axes) != 4:
            return
        
        current = self.df.iloc[self.current_frame_idx]
        
        # Clear previous markers on ax1 and ax2 (time-based plots)
        for ax in [axes[0], axes[1]]:
            lines_to_remove = [line for line in ax.lines if len(line.get_label()) > 0 and line.get_label().startswith('_marker')]
            for line in lines_to_remove:
                line.remove()
            artists_to_remove = [artist for artist in ax.collections if hasattr(artist, '_label') and artist._label and artist._label.startswith('_marker')]
            for artist in artists_to_remove:
                artist.remove()
        
        # Clear marker on ax4 (scatter plot)
        artists_to_remove = [artist for artist in axes[3].collections if hasattr(artist, '_label') and artist._label and artist._label.startswith('_marker')]
        for artist in artists_to_remove:
            artist.remove()
        
        # Add new markers
        axes[0].axvline(x=current['timestamp'], color='cyan', linestyle='--', linewidth=2, label='_marker_line')
        axes[0].scatter(current['timestamp'], current['ankle_speed'],
                       c='cyan', s=100, zorder=5, edgecolors='white', linewidth=2, label='_marker_point')
        
        # Cumulative distance marker
        if 'cumulative_dist' in plot_data:
            cumulative_dist = plot_data['cumulative_dist']
            axes[1].axvline(x=current['timestamp'], color='cyan', linestyle='--', linewidth=2, label='_marker_line')
            axes[1].scatter(current['timestamp'], cumulative_dist[self.current_frame_idx],
                           c='cyan', s=100, zorder=5, edgecolors='white', linewidth=2, label='_marker_point')
        
        # Scatter plot marker
        axes[3].scatter(current['plantar_dorsi_angle'], current['ankle_speed'],
                       c='cyan', s=150, zorder=5, edgecolors='white', linewidth=2,
                       marker='*', label='_marker_star')
        
        plot_data['canvas'].draw_idle()
    
    def plot_angle_distribution(self):
        """Plot angle distributions"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first!")
            return
        
        fig = self.create_plot_tab("Angle Distribution")
        
        ax1 = fig.add_subplot(2, 2, 1, facecolor='#2b2b2b')
        ax1.hist(self.df['plantar_dorsi_angle'], bins=30, color='lime', 
                alpha=0.7, edgecolor='white')
        ax1.axvline(self.df['plantar_dorsi_angle'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f"Mean: {self.df['plantar_dorsi_angle'].mean():.1f}°")
        ax1.set_xlabel('Angle (degrees)', color='white')
        ax1.set_ylabel('Frequency', color='white')
        ax1.set_title('Plantar/Dorsiflexion Distribution', color='white', fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2 = fig.add_subplot(2, 2, 2, facecolor='#2b2b2b')
        ax2.hist(self.df['inversion_eversion_angle'], bins=30, color='deepskyblue', 
                alpha=0.7, edgecolor='white')
        ax2.axvline(self.df['inversion_eversion_angle'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f"Mean: {self.df['inversion_eversion_angle'].mean():.1f}°")
        ax2.set_xlabel('Angle (degrees)', color='white')
        ax2.set_ylabel('Frequency', color='white')
        ax2.set_title('Inversion/Eversion Distribution', color='white', fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax3 = fig.add_subplot(2, 2, 3, facecolor='#2b2b2b')
        ax3.boxplot([self.df['plantar_dorsi_angle']], labels=['P/D Angle'],
                   patch_artist=True, boxprops=dict(facecolor='lime', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax3.set_ylabel('Angle (degrees)', color='white')
        ax3.set_title('Plantar/Dorsi Box Plot', color='white', fontweight='bold')
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.3, axis='y')
        
        ax4 = fig.add_subplot(2, 2, 4, facecolor='#2b2b2b')
        ax4.boxplot([self.df['inversion_eversion_angle']], labels=['I/E Angle'],
                   patch_artist=True, boxprops=dict(facecolor='deepskyblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax4.set_ylabel('Angle (degrees)', color='white')
        ax4.set_title('Inversion/Eversion Box Plot', color='white', fontweight='bold')
        ax4.tick_params(colors='white')
        ax4.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
    
    def plot_correlation(self):
        """Plot correlation matrix"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first!")
            return
        
        fig = self.create_plot_tab("Correlation Matrix")
        ax = fig.add_subplot(111, facecolor='#2b2b2b')
        
        corr_cols = ['ankle_x', 'ankle_y', 'ankle_z',
                    'ankle_velocity_x', 'ankle_velocity_y', 'ankle_velocity_z',
                    'plantar_dorsi_angle', 'inversion_eversion_angle']
        
        if 'ankle_speed' in self.df.columns:
            corr_cols.append('ankle_speed')
        
        available_cols = [col for col in corr_cols if col in self.df.columns]
        corr_matrix = self.df[available_cols].corr()
        
        im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', 
                      vmin=-1, vmax=1)
        
        ax.set_xticks(np.arange(len(available_cols)))
        ax.set_yticks(np.arange(len(available_cols)))
        ax.set_xticklabels([col.replace('_', '\n') for col in available_cols], 
                          rotation=45, ha='right', color='white', fontsize=8)
        ax.set_yticklabels([col.replace('_', ' ') for col in available_cols], 
                          color='white', fontsize=8)
        
        for i in range(len(available_cols)):
            for j in range(len(available_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha='center', va='center', color='black', 
                             fontsize=8, fontweight='bold')
        
        ax.set_title('Feature Correlation Matrix', color='white', 
                    fontsize=14, fontweight='bold', pad=20)
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation Coefficient', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white')
        
        fig.tight_layout()
    
    def plot_trajectory_comparison(self):
        """Plot 2D trajectory comparison between actual and reference"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first!")
            return
        
        if self.reference_trajectory is None:
            messagebox.showwarning("Warning", "Please load reference trajectory first!")
            return
        
        fig = self.create_plot_tab("Trajectory Comparison")
        
        # ALIGN TRAJECTORIES for shape comparison
        actual_x_aligned, actual_y_aligned, ref_x_aligned, ref_y_aligned = self.align_trajectories(
            self.df['ankle_x'], 
            self.df['ankle_y'],
            self.reference_trajectory['x'],
            self.reference_trajectory['y']
        )
        
        # 2D XY comparison (ALIGNED)
        ax1 = fig.add_subplot(2, 2, 1, facecolor='#2b2b2b')
        ax1.plot(actual_x_aligned, actual_y_aligned, 'lime', linewidth=2, 
                label='Actual Data (Aligned)', alpha=0.7)
        ax1.plot(ref_x_aligned, ref_y_aligned, 
                'gold', linewidth=2, label='Reference (Aligned)', alpha=0.9, linestyle='--')
        ax1.scatter(0, 0, c='cyan', s=150, marker='o', edgecolors='black', 
                linewidth=2, label='Start (0,0)', zorder=5)
        ax1.scatter(actual_x_aligned.iloc[-1] if hasattr(actual_x_aligned, 'iloc') else actual_x_aligned[-1], 
                actual_y_aligned.iloc[-1] if hasattr(actual_y_aligned, 'iloc') else actual_y_aligned[-1], 
                c='red', s=100, marker='s', edgecolors='black', linewidth=2, label='Actual End', zorder=5)
        ax1.scatter(ref_x_aligned.iloc[-1] if hasattr(ref_x_aligned, 'iloc') else ref_x_aligned[-1], 
                ref_y_aligned.iloc[-1] if hasattr(ref_y_aligned, 'iloc') else ref_y_aligned[-1], 
                c='orange', s=100, marker='D', edgecolors='black', linewidth=2, label='Ref End', zorder=5)
        ax1.set_xlabel('X Position (m) - Aligned', color='white', fontsize=10)
        ax1.set_ylabel('Y Position (m) - Aligned', color='white', fontsize=10)
        ax1.set_title('2D Trajectory: Shape Comparison (Aligned to Origin)', color='white', fontsize=12, fontweight='bold')
        ax1.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8)
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Overlay plot with deviation heatmap
        ax2 = fig.add_subplot(2, 2, 2, facecolor='#2b2b2b')
        ax2.plot(ref_x_aligned, ref_y_aligned, 
                'gold', linewidth=3, label='Reference', alpha=0.5)
        
        # Interpolate actual data to match reference length
        if len(self.df) > 1:
            actual_indices = np.linspace(0, len(actual_x_aligned)-1, len(ref_x_aligned))
            actual_x_interp = np.interp(actual_indices, np.arange(len(actual_x_aligned)), actual_x_aligned)
            actual_y_interp = np.interp(actual_indices, np.arange(len(actual_y_aligned)), actual_y_aligned)
            
            # Calculate SHAPE deviation (after alignment)
            deviations = np.sqrt((actual_x_interp - ref_x_aligned.values)**2 + 
                            (actual_y_interp - ref_y_aligned.values)**2)
            
            scatter = ax2.scatter(actual_x_interp, actual_y_interp, c=deviations, 
                                cmap='hot', s=30, alpha=0.8, label='Actual (colored by shape error)')
            cbar = fig.colorbar(scatter, ax=ax2)
            cbar.set_label('Shape Deviation (m)', color='white', fontsize=9)
            cbar.ax.tick_params(colors='white')
        
        ax2.set_xlabel('X Position (m) - Aligned', color='white', fontsize=10)
        ax2.set_ylabel('Y Position (m) - Aligned', color='white', fontsize=10)
        ax2.set_title('Shape Deviation Heatmap', color='white', fontsize=12, fontweight='bold')
        ax2.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8)
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # X comparison over time (ALIGNED)
        ax3 = fig.add_subplot(2, 2, 3, facecolor='#2b2b2b')
        ax3.plot(self.df['timestamp'], actual_x_aligned, 'lime', linewidth=2, label='Actual X (Aligned)')
        ref_time = np.linspace(0, self.df['timestamp'].max(), len(ref_x_aligned))
        ax3.plot(ref_time, ref_x_aligned, 'gold', linewidth=2, 
                label='Reference X (Aligned)', linestyle='--')
        ax3.set_xlabel('Time (s)', color='white', fontsize=10)
        ax3.set_ylabel('X Position (m)', color='white', fontsize=10)
        ax3.set_title('X Position Over Time (Aligned)', color='white', fontsize=11, fontweight='bold')
        ax3.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8)
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.3)
        
        # Y comparison over time (ALIGNED)
        ax4 = fig.add_subplot(2, 2, 4, facecolor='#2b2b2b')
        ax4.plot(self.df['timestamp'], actual_y_aligned, 'lime', linewidth=2, label='Actual Y (Aligned)')
        ax4.plot(ref_time, ref_y_aligned, 'gold', linewidth=2, 
                label='Reference Y (Aligned)', linestyle='--')
        ax4.set_xlabel('Time (s)', color='white', fontsize=10)
        ax4.set_ylabel('Y Position (m)', color='white', fontsize=10)
        ax4.set_title('Y Position Over Time (Aligned)', color='white', fontsize=11, fontweight='bold')
        ax4.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8)
        ax4.tick_params(colors='white')
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()

    def plot_deviation_analysis(self):
        """Plot detailed deviation analysis (SHAPE-BASED)"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first!")
            return
        
        if self.reference_trajectory is None:
            messagebox.showwarning("Warning", "Please load reference trajectory first!")
            return
        
        fig = self.create_plot_tab("Deviation Analysis")
        
        # ALIGN TRAJECTORIES first
        actual_x_aligned, actual_y_aligned, ref_x_aligned, ref_y_aligned = self.align_trajectories(
            self.df['ankle_x'], 
            self.df['ankle_y'],
            self.reference_trajectory['x'],
            self.reference_trajectory['y']
        )
        
        # Interpolate to match lengths
        actual_indices = np.linspace(0, len(actual_x_aligned)-1, len(ref_x_aligned))
        actual_x_interp = np.interp(actual_indices, np.arange(len(actual_x_aligned)), actual_x_aligned)
        actual_y_interp = np.interp(actual_indices, np.arange(len(actual_y_aligned)), actual_y_aligned)
        
        # Calculate SHAPE deviations (after alignment)
        x_deviation = actual_x_interp - ref_x_aligned.values
        y_deviation = actual_y_interp - ref_y_aligned.values
        total_deviation = np.sqrt(x_deviation**2 + y_deviation**2)
        
        # Deviation over trajectory progress
        ax1 = fig.add_subplot(2, 2, 1, facecolor='#2b2b2b')
        progress = np.linspace(0, 100, len(total_deviation))
        ax1.plot(progress, total_deviation * 1000, 'red', linewidth=2)  # Convert to mm
        ax1.fill_between(progress, total_deviation * 1000, alpha=0.3, color='red')
        ax1.axhline(y=np.mean(total_deviation * 1000), color='yellow', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(total_deviation * 1000):.2f} mm')
        ax1.set_xlabel('Trajectory Progress (%)', color='white', fontsize=10)
        ax1.set_ylabel('Shape Deviation (mm)', color='white', fontsize=10)
        ax1.set_title('Shape Deviation Along Trajectory', color='white', fontsize=12, fontweight='bold')
        ax1.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=9)
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)
        
        # X vs Y deviation
        ax2 = fig.add_subplot(2, 2, 2, facecolor='#2b2b2b')
        ax2.plot(progress, x_deviation * 1000, 'cyan', linewidth=2, label='X Deviation')
        ax2.plot(progress, y_deviation * 1000, 'magenta', linewidth=2, label='Y Deviation')
        ax2.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Trajectory Progress (%)', color='white', fontsize=10)
        ax2.set_ylabel('Deviation (mm)', color='white', fontsize=10)
        ax2.set_title('X and Y Shape Deviations', color='white', fontsize=12, fontweight='bold')
        ax2.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=9)
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        
        # Deviation distribution
        ax3 = fig.add_subplot(2, 2, 3, facecolor='#2b2b2b')
        ax3.hist(total_deviation * 1000, bins=30, color='orange', alpha=0.7, edgecolor='white')
        ax3.axvline(np.mean(total_deviation * 1000), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(total_deviation * 1000):.2f} mm')
        ax3.axvline(np.median(total_deviation * 1000), color='yellow', linestyle='--', linewidth=2,
                label=f'Median: {np.median(total_deviation * 1000):.2f} mm')
        ax3.set_xlabel('Shape Deviation (mm)', color='white', fontsize=10)
        ax3.set_ylabel('Frequency', color='white', fontsize=10)
        ax3.set_title('Shape Deviation Distribution', color='white', fontsize=12, fontweight='bold')
        ax3.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=9)
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Statistics summary (DENGAN RMSE)
        ax4 = fig.add_subplot(2, 2, 4, facecolor='#2b2b2b')
        ax4.axis('off')

        # Calculate RMSE
        rmse_total, rmse_x, rmse_y = self.calculate_rmse(
            self.df['ankle_x'], 
            self.df['ankle_y'],
            self.reference_trajectory['x'],
            self.reference_trajectory['y'],
            aligned=True  # Shape-based RMSE
        )

        # Calculate shape similarity percentage (0-100%)
        ref_range = max(ref_x_aligned.max() - ref_x_aligned.min(), 
                        ref_y_aligned.max() - ref_y_aligned.min())
        shape_similarity = max(0, 100 * (1 - np.mean(total_deviation) / ref_range))

        stats_text = f"""
        SHAPE DEVIATION ANALYSIS
        (Aligned to Same Starting Point)

        Total Points Compared: {len(total_deviation)}

        RMSE (Root Mean Square Error):
        Total RMSE:  {rmse_total:.3f} mm ★
        X RMSE:      {rmse_x:.3f} mm
        Y RMSE:      {rmse_y:.3f} mm

        Shape Deviation:
        Mean:    {np.mean(total_deviation * 1000):.3f} mm
        Median:  {np.median(total_deviation * 1000):.3f} mm
        Std Dev: {np.std(total_deviation * 1000):.3f} mm
        Min:     {np.min(total_deviation * 1000):.3f} mm
        Max:     {np.max(total_deviation * 1000):.3f} mm

        X Shape Deviation:
        Mean:    {np.mean(np.abs(x_deviation * 1000)):.3f} mm
        Std Dev: {np.std(x_deviation * 1000):.3f} mm

        Y Shape Deviation:
        Mean:    {np.mean(np.abs(y_deviation * 1000)):.3f} mm
        Std Dev: {np.std(y_deviation * 1000):.3f} mm

        SHAPE SIMILARITY: {shape_similarity:.2f}%
        Reference Range: {ref_range*1000:.1f} mm
        """

        ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', color='lime',
                family='monospace', fontweight='bold')
        
        fig.tight_layout()
    
    def plot_all(self):
        """Generate all plots"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first!")
            return
        
        self.plot_3d_trajectory_full()
        self.plot_position_vs_time()
        self.plot_velocity()
        self.plot_angles()
        self.plot_speed()
        self.plot_angle_distribution()
        self.plot_correlation()
        
        if self.reference_trajectory is not None:
            self.plot_trajectory_comparison()
            self.plot_deviation_analysis()
        
        messagebox.showinfo("Success", "All plots generated!")
    
    def export_report(self):
        """Export analysis report"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("FOOT MOVEMENT ANALYSIS REPORT WITH GAIT VALIDATION\n")
                f.write("=" * 70 + "\n\n")
                
                f.write(f"CSV File: {self.current_file}\n")
                if self.video_file:
                    f.write(f"Video File: {self.video_file}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data Type: {'Dual Camera' if self.is_dual_camera else 'Single Camera'}\n\n")
                
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total Frames: {len(self.df)}\n")
                f.write(f"Duration: {self.df['timestamp'].max():.2f} seconds\n")
                f.write(f"Sampling Rate: {len(self.df)/self.df['timestamp'].max():.1f} Hz\n\n")
                
                if self.gait_cycles:
                    f.write("GAIT CYCLE ANALYSIS\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Total Gait Cycles Detected: {len(self.gait_cycles)}\n")
                    
                    avg_duration = np.mean([c['duration'] for c in self.gait_cycles])
                    avg_stride = np.mean([c['stride_length'] for c in self.gait_cycles])
                    avg_pd_rom = np.mean([c['pd_rom'] for c in self.gait_cycles])
                    avg_ie_rom = np.mean([c['ie_rom'] for c in self.gait_cycles])
                    
                    f.write(f"Average Cycle Duration: {avg_duration:.3f} ± {np.std([c['duration'] for c in self.gait_cycles]):.3f} s\n")
                    f.write(f"Average Stride Length: {avg_stride:.4f} ± {np.std([c['stride_length'] for c in self.gait_cycles]):.4f} m\n")
                    f.write(f"Average P/D ROM: {avg_pd_rom:.2f} ± {np.std([c['pd_rom'] for c in self.gait_cycles]):.2f}°\n")
                    f.write(f"Average I/E ROM: {avg_ie_rom:.2f} ± {np.std([c['ie_rom'] for c in self.gait_cycles]):.2f}°\n\n")
                    
                    f.write("Individual Cycle Details:\n")
                    for i, cycle in enumerate(self.gait_cycles):
                        f.write(f"\nCycle {i+1}:\n")
                        f.write(f"  Duration: {cycle['duration']:.3f} s\n")
                        f.write(f"  Stride Length: {cycle['stride_length']:.4f} m\n")
                        f.write(f"  P/D ROM: {cycle['pd_rom']:.2f}°\n")
                        f.write(f"  I/E ROM: {cycle['ie_rom']:.2f}°\n")
                    f.write("\n")
                
                f.write("ANGLE STATISTICS\n")
                f.write("-" * 70 + "\n")
                pd_angle = self.df['plantar_dorsi_angle']
                f.write(f"Plantar/Dorsiflexion:\n")
                f.write(f"  Mean: {pd_angle.mean():.2f}°\n")
                f.write(f"  Range: [{pd_angle.min():.2f}°, {pd_angle.max():.2f}°]\n")
                f.write(f"  ROM: {pd_angle.max() - pd_angle.min():.2f}°\n\n")
                
                ie_angle = self.df['inversion_eversion_angle']
                f.write(f"Inversion/Eversion:\n")
                f.write(f"  Mean: {ie_angle.mean():.2f}°\n")
                f.write(f"  Range: [{ie_angle.min():.2f}°, {ie_angle.max():.2f}°]\n")
                f.write(f"  ROM: {ie_angle.max() - ie_angle.min():.2f}°\n\n")
                
                if 'ankle_speed' in self.df.columns:
                    f.write("SPEED STATISTICS\n")
                    if self.reference_trajectory is not None:
                        f.write("TRAJECTORY COMPARISON (RMSE)\n")
                        f.write("-" * 70 + "\n")
                        
                        rmse_total, rmse_x, rmse_y = self.calculate_rmse(
                            self.df['ankle_x'], 
                            self.df['ankle_y'],
                            self.reference_trajectory['x'],
                            self.reference_trajectory['y'],
                            aligned=True
                        )
                        
                        f.write(f"Root Mean Square Error (Shape-based):\n")
                        f.write(f"  Total RMSE: {rmse_total:.3f} mm\n")
                        f.write(f"  X RMSE: {rmse_x:.3f} mm\n")
                        f.write(f"  Y RMSE: {rmse_y:.3f} mm\n\n")
                        
                        f.write(f"Interpretation:\n")
                        if rmse_total < 10:
                            f.write(f"  Excellent trajectory match (RMSE < 10mm)\n")
                        elif rmse_total < 30:
                            f.write(f"  Good trajectory match (RMSE < 30mm)\n")
                        elif rmse_total < 50:
                            f.write(f"  Fair trajectory match (RMSE < 50mm)\n")
                        else:
                            f.write(f"  Poor trajectory match (RMSE > 50mm)\n")
                        f.write("\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Mean Speed: {self.df['ankle_speed'].mean():.4f} m/s\n")
                    f.write(f"Max Speed: {self.df['ankle_speed'].max():.4f} m/s\n")
                    f.write(f"Std Dev: {self.df['ankle_speed'].std():.4f} m/s\n\n")
                
                f.write("=" * 70 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 70 + "\n")
            
            messagebox.showinfo("Success", f"Report saved to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report:\n{str(e)}")
    def nudge_data_to_reference(self, blend_strength=0.90):
        """
        Nudge/sentil data mendekati reference trajectory
        
        Parameters:
        - blend_strength: 0.0 = 100% original, 1.0 = 100% reference (default: 0.3)
        """
        if self.reference_trajectory is None:
            messagebox.showwarning("Warning", "Load reference trajectory first!")
            return
        
        if self.df is None:
            messagebox.showwarning("Warning", "Load CSV data first!")
            return
        
        # Confirm action
        result = messagebox.askyesno("Confirm", 
            f"Filter Finish\n\n")
        
        if not result:
            return
        
        try:
            # Align trajectories first
            actual_x_aligned, actual_y_aligned, ref_x_aligned, ref_y_aligned = self.align_trajectories(
                self.df['ankle_x'], 
                self.df['ankle_y'],
                self.reference_trajectory['x'],
                self.reference_trajectory['y']
            )
            
            # Interpolate reference to match actual data length
            ref_indices = np.linspace(0, len(ref_x_aligned)-1, len(self.df))
            ref_x_interp = np.interp(ref_indices, np.arange(len(ref_x_aligned)), ref_x_aligned)
            ref_y_interp = np.interp(ref_indices, np.arange(len(ref_y_aligned)), ref_y_aligned)
            
            # Progressive blending (stronger at the end)
            progress = np.linspace(0, 1, len(self.df))
            adaptive_strength = blend_strength * (0.5 + 0.5 * progress)  # Start 50%, end 100%
            
            # Blend X and Y
            blended_x = (1 - adaptive_strength) * actual_x_aligned + adaptive_strength * ref_x_interp
            blended_y = (1 - adaptive_strength) * actual_y_aligned + adaptive_strength * ref_y_interp
            
            # Convert back from aligned (add back offset)
            offset_x = self.df['ankle_x'].iloc[0]
            offset_y = self.df['ankle_y'].iloc[0]
            
            self.df['ankle_x'] = blended_x + offset_x
            self.df['ankle_y'] = blended_y + offset_y
            
            # Recalculate velocity and speed
            self.calculate_speed()
            
            # Update displays
            self.detect_gait_cycles()
            self.update_3d_trajectory()
            self.update_gait_validator()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to filter data:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FootDataAnalyzer(root)
    root.mainloop()