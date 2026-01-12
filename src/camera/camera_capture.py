"""
Modul capture video dari webcam dan RealSense D435i
"""
import cv2
import numpy as np
import pyrealsense2 as rs
from typing import Optional, Tuple, Dict
import threading
import time


class WebcamCapture:
    """Kelas untuk capture dari webcam"""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, try_multiple_ids: bool = True):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.try_multiple_ids = try_multiple_ids
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.actual_camera_id = camera_id
        
    def initialize(self) -> bool:
        """Initialize webcam dengan timeout dan mencoba beberapa camera ID"""
        # List camera ID untuk dicoba
        camera_ids_to_try = [self.camera_id]
        if self.try_multiple_ids:
            # Coba beberapa ID umum (0, 1, 2)
            camera_ids_to_try.extend([i for i in range(3) if i != self.camera_id])
        
        for cam_id in camera_ids_to_try:
            try:
                print(f"Trying webcam ID: {cam_id}")
                
                # Coba dengan backend DirectShow untuk Windows (lebih cepat)
                cap = None
                try:
                    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
                except:
                    try:
                        cap = cv2.VideoCapture(cam_id)
                    except:
                        continue
                
                if cap is None or not cap.isOpened():
                    if cap:
                        cap.release()
                    continue
                
                # Set properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Kurangi buffer
                
                # Test read dengan timeout
                timeout = 2  # 2 detik timeout per camera
                
                def test_read():
                    try:
                        ret, _ = cap.read()
                        return ret
                    except:
                        return False
                
                # Test read di thread terpisah
                read_result = [False]
                read_thread = threading.Thread(target=lambda: read_result.__setitem__(0, test_read()), daemon=True)
                read_thread.start()
                read_thread.join(timeout=timeout)
                
                if read_thread.is_alive():
                    print(f"Webcam ID {cam_id} read timeout")
                    cap.release()
                    continue
                
                if read_result[0]:
                    # Berhasil!
                    self.cap = cap
                    self.actual_camera_id = cam_id
                    print(f"Webcam initialized successfully on ID: {cam_id}")
                    return True
                else:
                    cap.release()
                    
            except Exception as e:
                print(f"Error trying webcam ID {cam_id}: {e}")
                if 'cap' in locals() and cap is not None:
                    try:
                        cap.release()
                    except:
                        pass
                continue
        
        print("No working webcam found")
        return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame dari webcam"""
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def release(self):
        """Release webcam"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class RealSenseCapture:
    """Kelas untuk capture dari RealSense D435i dengan depth"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.align: Optional[rs.align] = None
        self.is_running = False
        
    def initialize(self) -> bool:
        """Initialize RealSense pipeline dengan timeout"""
        try:
            # Cek apakah ada RealSense device
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                print("No RealSense device found")
                return False
            
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure color stream
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Configure depth stream
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            
            # Start streaming dengan timeout
            try:
                # Try start dengan timeout 5 detik
                profile = self.pipeline.start(self.config)
                
                # Tunggu beberapa frame untuk stabilisasi
                for _ in range(5):
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    if frames:
                        break
                
                # Create align object to align depth frames to color frames
                align_to = rs.stream.color
                self.align = rs.align(align_to)
                
                # Get depth sensor and set options (skip jika terlalu lama)
                try:
                    depth_sensor = profile.get_device().first_depth_sensor()
                    if depth_sensor.supports(rs.option.visual_preset):
                        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy preset
                except:
                    pass  # Skip jika gagal, tidak critical
                
                return True
            except Exception as e:
                print(f"RealSense start timeout or error: {e}")
                if self.pipeline:
                    try:
                        self.pipeline.stop()
                    except:
                        pass
                return False
        except Exception as e:
            print(f"Error initializing RealSense: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[rs.depth_frame], Optional[np.ndarray]]:
        """
        Read frames dari RealSense
        
        Returns:
            (success, color_frame, depth_frame, depth_colormap)
        """
        if self.pipeline is None:
            return False, None, None, None
        
        try:
            # Wait for frames dengan timeout pendek
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            
            # Align depth frame to color frame
            aligned_frames = self.align.process(frames)
            
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                return False, None, None, None
            
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap on depth image
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            return True, color_image, aligned_depth_frame, depth_colormap
            
        except Exception as e:
            # Skip error untuk menghindari spam
            return False, None, None, None
    
    def get_depth_at_point(self, depth_frame: rs.depth_frame, x: int, y: int) -> float:
        """Get depth value at specific point"""
        if depth_frame is None:
            return 0.0
        return depth_frame.get_distance(x, y)
    
    def get_point_cloud(self, depth_frame: rs.depth_frame, color_frame: rs.video_frame, 
                       intrinsics: rs.intrinsics) -> Optional[np.ndarray]:
        """Convert depth frame to point cloud"""
        if depth_frame is None or color_frame is None:
            return None
        
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        return vertices
    
    def release(self):
        """Stop RealSense pipeline"""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
            self.config = None
            self.align = None


class SynchronizedCapture:
    """Kelas untuk capture simultan dari webcam dan RealSense"""
    
    def __init__(self, webcam_id: int = 0, width: int = 640, height: int = 480):
        self.webcam = WebcamCapture(webcam_id, width, height)
        self.realsense = RealSenseCapture(width, height)
        self.is_running = False
        self.lock = threading.Lock()
        
        # Buffer untuk frames
        self.webcam_frame: Optional[np.ndarray] = None
        self.rs_color_frame: Optional[np.ndarray] = None
        self.rs_depth_frame: Optional[rs.depth_frame] = None
        self.rs_depth_colormap: Optional[np.ndarray] = None
        
    def initialize(self) -> Dict[str, bool]:
        """Initialize kedua kamera dengan prioritas webcam dulu"""
        results = {'webcam': False, 'realsense': False}
        
        # Initialize webcam dengan timeout
        print("Initializing webcam...")
        start_time = time.time()
        timeout = 5  # 5 detik timeout untuk webcam
        
        def init_webcam():
            try:
                return self.webcam.initialize()
            except:
                return False
        
        # Jalankan di thread dengan timeout
        webcam_result = [False]
        webcam_thread = threading.Thread(target=lambda: webcam_result.__setitem__(0, init_webcam()), daemon=True)
        webcam_thread.start()
        webcam_thread.join(timeout=timeout)
        
        if webcam_thread.is_alive():
            print("Webcam initialization timeout - skipping")
            results['webcam'] = False
            # Release webcam jika masih mencoba initialize
            try:
                if self.webcam.cap:
                    self.webcam.cap.release()
                    self.webcam.cap = None
            except:
                pass
        else:
            results['webcam'] = webcam_result[0]
            print(f"Webcam: {'OK' if results['webcam'] else 'Failed'} (took {time.time() - start_time:.2f}s)")
        
        # Initialize RealSense dengan timeout (bisa lama)
        print("Initializing RealSense...")
        start_time = time.time()
        timeout = 10  # 10 detik timeout
        
        def init_realsense():
            try:
                return self.realsense.initialize()
            except:
                return False
        
        # Jalankan di thread dengan timeout
        rs_result = [False]
        rs_thread = threading.Thread(target=lambda: rs_result.__setitem__(0, init_realsense()), daemon=True)
        rs_thread.start()
        rs_thread.join(timeout=timeout)
        
        if rs_thread.is_alive():
            print("RealSense initialization timeout - skipping")
            results['realsense'] = False
            # Stop RealSense jika masih mencoba initialize
            try:
                if self.realsense.pipeline:
                    self.realsense.pipeline.stop()
            except:
                pass
        else:
            results['realsense'] = rs_result[0]
            print(f"RealSense: {'OK' if results['realsense'] else 'Failed'}")
        
        elapsed = time.time() - start_time
        print(f"Initialization completed in {elapsed:.2f} seconds")
        
        return results
    
    def start_capture(self):
        """Start capture threads"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start webcam capture thread
        webcam_thread = threading.Thread(target=self._webcam_capture_loop, daemon=True)
        webcam_thread.start()
        
        # Start RealSense capture thread
        rs_thread = threading.Thread(target=self._realsense_capture_loop, daemon=True)
        rs_thread.start()
    
    def _webcam_capture_loop(self):
        """Loop untuk capture webcam"""
        while self.is_running:
            ret, frame = self.webcam.read()
            if ret:
                with self.lock:
                    self.webcam_frame = frame.copy()
            time.sleep(0.01)  # ~100 FPS untuk thread
    
    def _realsense_capture_loop(self):
        """Loop untuk capture RealSense"""
        while self.is_running:
            ret, color, depth, depth_colormap = self.realsense.read()
            if ret:
                with self.lock:
                    self.rs_color_frame = color.copy() if color is not None else None
                    self.rs_depth_frame = depth
                    self.rs_depth_colormap = depth_colormap.copy() if depth_colormap is not None else None
            time.sleep(0.01)
    
    def get_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """Get latest frames dari kedua kamera"""
        with self.lock:
            frames = {
                'webcam': None,
                'realsense_color': None,
                'realsense_depth': None,
                'realsense_depth_frame': None
            }
            
            if self.webcam_frame is not None:
                frames['webcam'] = self.webcam_frame.copy()
            
            if self.rs_color_frame is not None:
                frames['realsense_color'] = self.rs_color_frame.copy()
            
            if self.rs_depth_colormap is not None:
                frames['realsense_depth'] = self.rs_depth_colormap.copy()
            
            frames['realsense_depth_frame'] = self.rs_depth_frame
            
            return frames
    
    def stop_capture(self):
        """Stop capture threads"""
        self.is_running = False
        time.sleep(0.1)  # Wait for threads to finish
    
    def release(self):
        """Release semua kamera"""
        self.stop_capture()
        self.webcam.release()
        self.realsense.release()

