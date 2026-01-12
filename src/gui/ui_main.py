"""
UI utama aplikasi motion capture
"""
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
                             QTextEdit, QMessageBox, QTabWidget, QCheckBox, QProgressDialog)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
import os
import time


class VideoDisplayWidget(QLabel):
    """Widget untuk menampilkan video"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setText("No video")
        self.setStyleSheet("background-color: black; color: white;")
    
    def set_frame(self, frame: np.ndarray):
        """Set frame untuk ditampilkan"""
        if frame is None:
            return
        
        h, w = frame.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale untuk fit widget
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)


class CalibrationWidget(QWidget):
    """Widget untuk kalibrasi kamera"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Settings
        settings_group = QGroupBox("Settings Kalibrasi ChArUco")
        settings_layout = QGridLayout()
        
        settings_layout.addWidget(QLabel("Squares X (Columns):"), 0, 0)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(3, 15)
        self.cols_spin.setValue(7)
        settings_layout.addWidget(self.cols_spin, 0, 1)
        
        settings_layout.addWidget(QLabel("Squares Y (Rows):"), 1, 0)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(3, 15)
        self.rows_spin.setValue(5)
        settings_layout.addWidget(self.rows_spin, 1, 1)
        
        settings_layout.addWidget(QLabel("Square Length (mm):"), 2, 0)
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(1.0, 100.0)
        self.square_size_spin.setValue(40.0)
        settings_layout.addWidget(self.square_size_spin, 2, 1)
        
        settings_layout.addWidget(QLabel("Marker Length (mm):"), 3, 0)
        self.marker_size_spin = QDoubleSpinBox()
        self.marker_size_spin.setRange(1.0, 100.0)
        self.marker_size_spin.setValue(30.0)
        settings_layout.addWidget(self.marker_size_spin, 3, 1)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Auto-capture settings
        auto_group = QGroupBox("Auto Capture")
        auto_layout = QHBoxLayout()
        
        self.auto_capture_cb = QCheckBox("Enable Auto Capture")
        self.auto_capture_cb.setChecked(False)
        auto_layout.addWidget(self.auto_capture_cb)
        
        auto_layout.addWidget(QLabel("Interval (detik):"))
        self.auto_interval_spin = QSpinBox()
        self.auto_interval_spin.setRange(1, 10)
        self.auto_interval_spin.setValue(2)
        auto_layout.addWidget(self.auto_interval_spin)
        
        self.auto_capture_label = QLabel("0 images captured")
        auto_layout.addWidget(self.auto_capture_label)
        
        auto_group.setLayout(auto_layout)
        layout.addWidget(auto_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.capture_btn = QPushButton("Capture Image (Manual)")
        self.calibrate_btn = QPushButton("Kalibrasi")
        self.save_btn = QPushButton("Save Calibration")
        self.load_btn = QPushButton("Load Calibration")
        self.reset_btn = QPushButton("Reset")
        
        btn_layout.addWidget(self.capture_btn)
        btn_layout.addWidget(self.calibrate_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.reset_btn)
        
        layout.addLayout(btn_layout)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(80)
        status_layout.addWidget(self.status_text)
        
        # Calibration status
        self.calib_status_label = QLabel("Kalibrasi: Belum di-load")
        self.calib_status_label.setStyleSheet("font-weight: bold; color: red;")
        status_layout.addWidget(self.calib_status_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        self.setLayout(layout)
    
    def log_status(self, message: str):
        """Log status message"""
        self.status_text.append(message)


class CaptureWidget(QWidget):
    """Widget untuk capture dan motion capture"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Display area
        display_layout = QHBoxLayout()
        
        self.webcam_display = VideoDisplayWidget()
        self.webcam_display.setText("Webcam")
        display_layout.addWidget(self.webcam_display)
        
        self.realsense_display = VideoDisplayWidget()
        self.realsense_display.setText("RealSense Color")
        display_layout.addWidget(self.realsense_display)
        
        self.depth_display = VideoDisplayWidget()
        self.depth_display.setText("RealSense Depth")
        display_layout.addWidget(self.depth_display)
        
        layout.addLayout(display_layout)
        
        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Capture")
        self.stop_btn = QPushButton("Stop Capture")
        self.record_btn = QPushButton("Start Recording")
        self.pose_estimation_cb = QCheckBox("Enable Pose Estimation")
        self.pose_estimation_cb.setChecked(True)
        
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.record_btn)
        controls_layout.addWidget(self.pose_estimation_cb)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Info
        self.info_label = QLabel("Status: Stopped")
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
    
    def update_info(self, text: str):
        """Update info label"""
        self.info_label.setText(text)


class CameraInitThread(QThread):
    """Thread untuk inisialisasi kamera agar tidak freeze UI"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, capture):
        super().__init__()
        self.capture = capture
    
    def run(self):
        """Run inisialisasi di thread terpisah"""
        try:
            self.progress.emit("Menginisialisasi webcam...")
            results = self.capture.initialize()
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main window aplikasi"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FreeMoCap - Motion Capture System")
        self.setGeometry(100, 100, 1400, 800)
        
        # Import modules
        from camera.camera_capture import SynchronizedCapture
        from calibration.camera_calibration import CameraCalibrator, RealSenseCalibrator
        from pose.pose_estimation import PoseEstimator
        
        self.capture = SynchronizedCapture()
        # Calibrators akan di-init setelah UI dibuat untuk mendapatkan settings
        self.webcam_calibrator = None
        self.rs_calibrator = None
        self.pose_estimator = PoseEstimator()
        
        # Storage untuk kalibrasi yang sudah di-load
        self.webcam_camera_matrix = None
        self.webcam_dist_coeffs = None
        self.rs_camera_matrix = None
        self.rs_dist_coeffs = None
        
        # State
        self.is_capturing = False
        self.is_recording = False
        self.recorded_data = []
        self.init_thread = None
        
        # Timer untuk update display
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        
        # Timer untuk auto-capture kalibrasi
        self.auto_calib_timer = QTimer()
        self.auto_calib_timer.timeout.connect(self.auto_capture_calibration)
        self.last_calib_capture_time = 0
        
        self.init_ui()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Tabs
        tabs = QTabWidget()
        
        # Calibration tab
        self.calibration_widget = CalibrationWidget()
        self.calibration_widget.capture_btn.clicked.connect(self.capture_calibration_image)
        self.calibration_widget.calibrate_btn.clicked.connect(self.calibrate_cameras)
        self.calibration_widget.save_btn.clicked.connect(self.save_calibration)
        self.calibration_widget.load_btn.clicked.connect(self.load_calibration)
        self.calibration_widget.reset_btn.clicked.connect(self.reset_calibration)
        self.calibration_widget.auto_capture_cb.stateChanged.connect(self.toggle_auto_capture)
        # Update calibrators saat settings berubah (tidak auto-update, user harus reset manual)
        tabs.addTab(self.calibration_widget, "Kalibrasi")
        
        # Init calibrators setelah UI dibuat
        self.init_calibrators()
        
        # Capture tab
        self.capture_widget = CaptureWidget()
        self.capture_widget.start_btn.clicked.connect(self.start_capture)
        self.capture_widget.stop_btn.clicked.connect(self.stop_capture)
        self.capture_widget.record_btn.clicked.connect(self.toggle_recording)
        tabs.addTab(self.capture_widget, "Motion Capture")
        
        # Visualize tab
        from visualization.visualize_gui import VisualizeWidget
        self.visualize_widget = VisualizeWidget()
        tabs.addTab(self.visualize_widget, "Visualize")
        
        layout.addWidget(tabs)
        central_widget.setLayout(layout)
    
    def init_calibrators(self):
        """Initialize calibrators dengan settings dari UI"""
        from calibration.camera_calibration import CameraCalibrator, RealSenseCalibrator
        
        if hasattr(self, 'calibration_widget'):
            squares_x = self.calibration_widget.cols_spin.value()
            squares_y = self.calibration_widget.rows_spin.value()
            square_length = self.calibration_widget.square_size_spin.value()
            marker_length = self.calibration_widget.marker_size_spin.value()
        else:
            squares_x = 7
            squares_y = 5
            square_length = 40.0
            marker_length = 30.0
        
        self.webcam_calibrator = CameraCalibrator(squares_x, squares_y, square_length, marker_length)
        self.rs_calibrator = RealSenseCalibrator(squares_x, squares_y, square_length, marker_length)
    
    def capture_calibration_image(self, auto_mode=False):
        """Capture image untuk kalibrasi"""
        if not self.is_capturing:
            if not auto_mode:
                QMessageBox.warning(self, "Warning", "Start capture terlebih dahulu!")
            return False
        
        frames = self.capture.get_frames()
        captured = False
        
        # Capture untuk webcam
        if frames['webcam'] is not None:
            found = self.webcam_calibrator.add_calibration_image(frames['webcam'])
            if found:
                count = len(self.webcam_calibrator.all_charuco_corners) if hasattr(self.webcam_calibrator, 'all_charuco_corners') else 0
                self.calibration_widget.log_status(f"Webcam: ChArUco board ditemukan! Total: {count}")
                captured = True
                self.update_calibration_count()
            elif not auto_mode:
                self.calibration_widget.log_status("Webcam: ChArUco board tidak ditemukan")
        
        # Capture untuk RealSense
        if frames['realsense_color'] is not None:
            found = self.rs_calibrator.add_calibration_image(frames['realsense_color'])
            if found:
                count = len(self.rs_calibrator.all_charuco_corners) if hasattr(self.rs_calibrator, 'all_charuco_corners') else 0
                self.calibration_widget.log_status(f"RealSense: ChArUco board ditemukan! Total: {count}")
                captured = True
                self.update_calibration_count()
            elif not auto_mode:
                self.calibration_widget.log_status("RealSense: ChArUco board tidak ditemukan")
        
        return captured
    
    def update_calibration_count(self):
        """Update label jumlah gambar kalibrasi"""
        webcam_count = len(self.webcam_calibrator.all_charuco_corners) if hasattr(self.webcam_calibrator, 'all_charuco_corners') else 0
        rs_count = len(self.rs_calibrator.all_charuco_corners) if hasattr(self.rs_calibrator, 'all_charuco_corners') else 0
        total = max(webcam_count, rs_count)
        self.calibration_widget.auto_capture_label.setText(f"{total} images captured")
    
    def toggle_auto_capture(self, state):
        """Toggle auto-capture untuk kalibrasi"""
        if state == Qt.Checked:
            if not self.is_capturing:
                QMessageBox.warning(self, "Warning", "Start capture terlebih dahulu!")
                self.calibration_widget.auto_capture_cb.setChecked(False)
                return
            
            interval = self.calibration_widget.auto_interval_spin.value() * 1000  # Convert to milliseconds
            self.auto_calib_timer.start(interval)
            self.calibration_widget.log_status("Auto-capture enabled")
        else:
            self.auto_calib_timer.stop()
            self.calibration_widget.log_status("Auto-capture disabled")
    
    def auto_capture_calibration(self):
        """Auto-capture untuk kalibrasi (dipanggil oleh timer)"""
        if not self.is_capturing:
            self.auto_calib_timer.stop()
            self.calibration_widget.auto_capture_cb.setChecked(False)
            return
        
        # Capture hanya jika ChArUco board terdeteksi
        self.capture_calibration_image(auto_mode=True)
    
    def calibrate_cameras(self):
        """Lakukan kalibrasi"""
        try:
            frames = self.capture.get_frames()
            
            # Kalibrasi webcam
            webcam_count = len(self.webcam_calibrator.all_charuco_corners) if hasattr(self.webcam_calibrator, 'all_charuco_corners') else 0
            if webcam_count >= 5:
                if frames['webcam'] is not None:
                    h, w = frames['webcam'].shape[:2]
                    ret, mtx, dist, rvecs, tvecs = self.webcam_calibrator.calibrate((w, h))
                    self.calibration_widget.log_status(f"Webcam kalibrasi selesai! Reprojection Error: {ret:.4f}")
            
            # Kalibrasi RealSense
            rs_count = len(self.rs_calibrator.all_charuco_corners) if hasattr(self.rs_calibrator, 'all_charuco_corners') else 0
            if rs_count >= 5:
                if frames['realsense_color'] is not None:
                    h, w = frames['realsense_color'].shape[:2]
                    ret, mtx, dist, rvecs, tvecs = self.rs_calibrator.calibrate((w, h))
                    self.calibration_widget.log_status(f"RealSense kalibrasi selesai! Reprojection Error: {ret:.4f}")
            
            if webcam_count < 10 and rs_count < 10:
                QMessageBox.warning(self, "Warning", "Minimal 10 gambar kalibrasi diperlukan untuk ChArUco!")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saat kalibrasi: {str(e)}")
    
    def save_calibration(self):
        """Save kalibrasi ke file"""
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Calibration", "", "JSON Files (*.json)")
        if filepath:
            try:
                frames = self.capture.get_frames()
                if frames['webcam'] is not None:
                    h, w = frames['webcam'].shape[:2]
                    ret, mtx, dist, _, _ = self.webcam_calibrator.calibrate((w, h))
                    self.webcam_calibrator.save_calibration(filepath.replace('.json', '_webcam.json'), mtx, dist, (w, h))
                
                if frames['realsense_color'] is not None:
                    h, w = frames['realsense_color'].shape[:2]
                    ret, mtx, dist, _, _ = self.rs_calibrator.calibrate((w, h))
                    self.rs_calibrator.save_calibration(filepath.replace('.json', '_realsense.json'), mtx, dist, (w, h))
                
                QMessageBox.information(self, "Success", "Kalibrasi berhasil disimpan!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saat save: {str(e)}")
    
    def load_calibration(self):
        """Load kalibrasi dari file (bisa load kedua file sekaligus atau salah satu)"""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        
        # Option 1: Load kedua file sekaligus
        reply = QMessageBox.question(
            self, 
            'Load Calibration', 
            'Pilih cara load:\n\nYes = Load kedua file (webcam + realsense)\nNo = Load satu file saja',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # Load kedua file sekaligus
            webcam_file, _ = QFileDialog.getOpenFileName(
                self, "Load Webcam Calibration", "", "JSON Files (*.json)"
            )
            if not webcam_file:
                return
            
            realsense_file, _ = QFileDialog.getOpenFileName(
                self, "Load RealSense Calibration", "", "JSON Files (*.json)"
            )
            if not realsense_file:
                return
            
            try:
                # Load webcam
                mtx_wc, dist_wc, size_wc = self.webcam_calibrator.load_calibration(webcam_file)
                self.webcam_camera_matrix = mtx_wc
                self.webcam_dist_coeffs = dist_wc
                self.calibration_widget.log_status(f"Webcam kalibrasi loaded: {size_wc}")
                
                # Load realsense
                mtx_rs, dist_rs, size_rs = self.rs_calibrator.load_calibration(realsense_file)
                self.rs_camera_matrix = mtx_rs
                self.rs_dist_coeffs = dist_rs
                self.calibration_widget.log_status(f"RealSense kalibrasi loaded: {size_rs}")
                
                # Update status label
                self.calibration_widget.calib_status_label.setText(
                    "Kalibrasi: Webcam ✓ | RealSense ✓"
                )
                self.calibration_widget.calib_status_label.setStyleSheet("font-weight: bold; color: green;")
                
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Kalibrasi berhasil dimuat!\n\nWebcam: {size_wc}\nRealSense: {size_rs}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saat load: {str(e)}")
        else:
            # Load satu file saja
            filepath, _ = QFileDialog.getOpenFileName(self, "Load Calibration", "", "JSON Files (*.json)")
            if filepath:
                try:
                    if 'webcam' in filepath.lower():
                        mtx, dist, size = self.webcam_calibrator.load_calibration(filepath)
                        self.webcam_camera_matrix = mtx
                        self.webcam_dist_coeffs = dist
                        self.calibration_widget.log_status(f"Webcam kalibrasi loaded: {size}")
                        # Update status
                        rs_status = "✓" if self.rs_camera_matrix is not None else "✗"
                        self.calibration_widget.calib_status_label.setText(
                            f"Kalibrasi: Webcam ✓ | RealSense {rs_status}"
                        )
                        self.calibration_widget.calib_status_label.setStyleSheet("font-weight: bold; color: orange;")
                        QMessageBox.information(self, "Success", f"Webcam kalibrasi berhasil dimuat!\nSize: {size}")
                    elif 'realsense' in filepath.lower():
                        mtx, dist, size = self.rs_calibrator.load_calibration(filepath)
                        self.rs_camera_matrix = mtx
                        self.rs_dist_coeffs = dist
                        self.calibration_widget.log_status(f"RealSense kalibrasi loaded: {size}")
                        # Update status
                        wc_status = "✓" if self.webcam_camera_matrix is not None else "✗"
                        self.calibration_widget.calib_status_label.setText(
                            f"Kalibrasi: Webcam {wc_status} | RealSense ✓"
                        )
                        self.calibration_widget.calib_status_label.setStyleSheet("font-weight: bold; color: orange;")
                        QMessageBox.information(self, "Success", f"RealSense kalibrasi berhasil dimuat!\nSize: {size}")
                    else:
                        # Try to detect based on file content
                        import json
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            # Assume webcam if can't determine
                            mtx, dist, size = self.webcam_calibrator.load_calibration(filepath)
                            self.webcam_camera_matrix = mtx
                            self.webcam_dist_coeffs = dist
                            self.calibration_widget.log_status(f"Kalibrasi loaded (as webcam): {size}")
                            # Update status
                            rs_status = "✓" if self.rs_camera_matrix is not None else "✗"
                            self.calibration_widget.calib_status_label.setText(
                                f"Kalibrasi: Webcam ✓ | RealSense {rs_status}"
                            )
                            self.calibration_widget.calib_status_label.setStyleSheet("font-weight: bold; color: orange;")
                            QMessageBox.information(self, "Success", f"Kalibrasi berhasil dimuat sebagai webcam!\nSize: {size}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error saat load: {str(e)}")
    
    def reset_calibration(self):
        """Reset kalibrasi"""
        self.webcam_calibrator.reset()
        self.rs_calibrator.reset()
        self.update_calibration_count()
        self.calibration_widget.log_status("Kalibrasi direset")
    
    def start_capture(self):
        """Start capture dari kamera"""
        # Disable button saat inisialisasi
        self.capture_widget.start_btn.setEnabled(False)
        self.capture_widget.update_info("Status: Menginisialisasi kamera...")
        
        # Buat progress dialog
        self.progress = QProgressDialog("Menginisialisasi kamera...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)  # Tidak bisa dibatalkan
        self.progress.setMinimumDuration(0)
        self.progress.show()
        QApplication.processEvents()  # Update UI
        
        # Inisialisasi di thread terpisah
        self.init_thread = CameraInitThread(self.capture)
        self.init_thread.finished.connect(self.on_camera_initialized)
        self.init_thread.error.connect(self.on_camera_init_error)
        self.init_thread.progress.connect(self.on_progress_update)
        self.init_thread.start()
    
    def on_progress_update(self, message):
        """Update progress message"""
        if hasattr(self, 'progress'):
            self.progress.setLabelText(message)
            QApplication.processEvents()
    
    def on_camera_initialized(self, results):
        """Callback saat kamera selesai diinisialisasi"""
        self.progress.close()
        
        if not results['webcam'] and not results['realsense']:
            QMessageBox.critical(self, "Error", 
                               "Tidak ada kamera yang bisa diakses!\n\n"
                               "Pastikan:\n"
                               "- Webcam terhubung dan tidak digunakan aplikasi lain\n"
                               "- RealSense terhubung ke USB 3.0 (jika digunakan)")
            self.capture_widget.start_btn.setEnabled(True)
            return
        
        # Start capture threads
        try:
            self.capture.start_capture()
            
            # Tunggu sebentar untuk frames pertama
            time.sleep(0.3)
            
            self.is_capturing = True
            self.timer.start(33)  # ~30 FPS
            self.capture_widget.start_btn.setEnabled(True)
            
            # Update status
            status_msg = "Status: Capturing"
            if results['webcam']:
                status_msg += " (Webcam OK)"
            else:
                status_msg += " (Webcam: Tidak tersedia)"
            if results['realsense']:
                status_msg += " (RealSense OK)"
            else:
                status_msg += " (RealSense: Tidak tersedia)"
            self.capture_widget.update_info(status_msg)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saat start capture: {str(e)}")
            self.capture_widget.start_btn.setEnabled(True)
            self.capture_widget.update_info("Status: Error - Coba lagi")
    
    def on_camera_init_error(self, error_msg):
        """Callback saat ada error inisialisasi"""
        self.progress.close()
        QMessageBox.critical(self, "Error", f"Error saat inisialisasi kamera:\n{error_msg}")
        self.capture_widget.start_btn.setEnabled(True)
        self.capture_widget.update_info("Status: Error - Coba lagi")
    
    def stop_capture(self):
        """Stop capture"""
        self.is_capturing = False
        self.timer.stop()
        self.auto_calib_timer.stop()  # Stop auto-capture juga
        if hasattr(self.calibration_widget, 'auto_capture_cb'):
            self.calibration_widget.auto_capture_cb.setChecked(False)
        self.capture.stop_capture()
        self.capture.release()
        self.capture_widget.update_info("Status: Stopped")
    
    def toggle_recording(self):
        """Toggle recording"""
        if not self.is_capturing:
            QMessageBox.warning(self, "Warning", "Start capture terlebih dahulu!")
            return
        
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.recorded_data = []
            self.capture_widget.record_btn.setText("Stop Recording")
            self.capture_widget.update_info("Status: Recording...")
        else:
            self.save_recording()
            self.capture_widget.record_btn.setText("Start Recording")
            self.capture_widget.update_info("Status: Capturing...")
    
    def save_recording(self):
        """Save recorded data"""
        if not self.recorded_data:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Recording", "", "JSON Files (*.json)")
        if filepath:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.recorded_data, f, indent=2)
            QMessageBox.information(self, "Success", f"Recording disimpan: {len(self.recorded_data)} frames")
    
    def update_display(self):
        """Update display dengan frames terbaru"""
        if not self.is_capturing:
            return
        
        frames = self.capture.get_frames()
        
        # Check apakah sedang di tab kalibrasi
        current_tab = self.centralWidget().findChild(QTabWidget)
        if current_tab:
            current_index = current_tab.currentIndex()
            is_calibration_tab = (current_index == 0)  # Tab pertama adalah kalibrasi
            
            # Update webcam display
            if frames['webcam'] is not None:
                webcam_frame = frames['webcam'].copy()
                
                # Jika di tab kalibrasi, tampilkan deteksi ChArUco
                if is_calibration_tab and hasattr(self, 'webcam_calibrator'):
                    webcam_frame = self.webcam_calibrator.draw_charuco_board(webcam_frame)
                
                self.capture_widget.webcam_display.set_frame(webcam_frame)
            
            # Update RealSense color display
            if frames['realsense_color'] is not None:
                rs_frame = frames['realsense_color'].copy()
                
                # Jika di tab kalibrasi, tampilkan deteksi ChArUco
                if is_calibration_tab and hasattr(self, 'rs_calibrator'):
                    rs_frame = self.rs_calibrator.draw_charuco_board(rs_frame)
                # Pose estimation jika enabled dan di tab motion capture
                elif self.capture_widget.pose_estimation_cb.isChecked():
                    landmarks = self.pose_estimator.process(rs_frame)
                    if landmarks:
                        rs_frame = self.pose_estimator.draw_landmarks(rs_frame, landmarks)
                        
                        # Record jika sedang recording
                        if self.is_recording and frames['realsense_depth_frame'] is not None:
                            try:
                                import pyrealsense2 as rs
                                if self.capture.realsense.pipeline is not None:
                                    profile = self.capture.realsense.pipeline.get_active_profile()
                                    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
                                    intrinsics = color_profile.get_intrinsics()
                                    
                                    data = self.pose_estimator.landmarks_to_dict(
                                        landmarks, 
                                        rs_frame.shape,
                                        frames['realsense_depth_frame'],
                                        intrinsics
                                    )
                                    self.recorded_data.append(data)
                            except Exception as e:
                                print(f"Error recording data: {e}")
                
                self.capture_widget.realsense_display.set_frame(rs_frame)
            
            # Update depth display
            if frames['realsense_depth'] is not None:
                self.capture_widget.depth_display.set_frame(frames['realsense_depth'])
    
    def closeEvent(self, event):
        """Cleanup saat window ditutup"""
        self.stop_capture()
        self.pose_estimator.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

