# Dual Camera Foot Tracking System

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start Guide](#quick-start-guide)
- [Program Details](#program-details)
  - [1. Camera Calibration Tool](#1-camera-calibration-tool)
  - [2. Dual Camera Foot Tracker](#2-dual-camera-foot-tracker)
  - [3. Data Analyzer](#3-data-analyzer)
- [Workflow](#workflow)
- [Data Output](#data-output)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)
- [Citations](#citations)

---

## 🎯 Overview

A comprehensive **3D foot movement tracking and analysis system** using dual-camera setup (Intel RealSense D435i + Webcam/GoPro) with MediaPipe pose estimation and advanced data filtering. The system is designed for **gait analysis**, **biomechanical research**, and **clinical applications**.

### Key Capabilities
- **3D triangulation** for accurate spatial position tracking
- **Real-time pose estimation** using MediaPipe
- **Advanced filtering** (Kalman + Butterworth) for noise reduction
- **Gait cycle detection** and validation
- **Reference trajectory comparison** with RMSE calculation
- **Professional data visualization** with synchronized video playback

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DUAL CAMERA SETUP                         │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │  RealSense D435i │          │  Webcam/GoPro    │        │
│  │  (Depth + RGB)   │          │  (RGB Only)      │        │
│  └────────┬─────────┘          └────────┬─────────┘        │
│           │                              │                  │
│           └──────────┬──────────────────┘                  │
│                      │                                      │
│              ┌───────▼────────┐                            │
│              │  MediaPipe     │                            │
│              │  Pose Detection│                            │
│              └───────┬────────┘                            │
│                      │                                      │
│              ┌───────▼────────┐                            │
│              │  3D Triangulation                           │
│              │  (Stereo Vision) │                          │
│              └───────┬────────┘                            │
│                      │                                      │
│         ┌────────────▼─────────────┐                       │
│         │  Data Recording          │                       │
│         │  - CSV (positions/angles)│                       │
│         │  - MP4 (synchronized)    │                       │
│         └────────────┬─────────────┘                       │
│                      │                                      │
│         ┌────────────▼─────────────┐                       │
│         │  Post-Processing         │                       │
│         │  - Filtering (Kalman)    │                       │
│         │  - Gait Cycle Detection  │                       │
│         │  - RMSE Comparison       │                       │
│         └──────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Camera Calibration Tool
- ✅ **Dual pattern support**: Checkerboard & ChArUco
- ✅ **Interactive calibration**: Real-time pattern detection
- ✅ **Automatic calculation**: Intrinsic & extrinsic parameters
- ✅ **Stereo baseline estimation**: Camera separation distance
- ✅ **Export calibration**: JSON format for reusability

### Foot Tracker
- ✅ **3D position tracking**: Ankle, heel, toe landmarks
- ✅ **Angle measurement**: Plantar/dorsiflexion, inversion/eversion
- ✅ **Velocity & speed**: Real-time calculation
- ✅ **Video recording**: Synchronized with CSV data
- ✅ **Live visualization**: 3D trajectory + time-series plots

### Data Analyzer
- ✅ **Advanced filtering**: Outlier removal, Kalman filter, Butterworth filter
- ✅ **Gait cycle validator**: Automatic detection & validation
- ✅ **Reference comparison**: RMSE analysis vs ideal trajectory
- ✅ **Multi-plot analysis**: Position, velocity, angles, speed
- ✅ **Video synchronization**: Frame-by-frame analysis
- ✅ **Report export**: Comprehensive text reports

---

## 💻 Requirements

### Hardware
- **Intel RealSense D435i** (depth camera)
- **Webcam or GoPro** (secondary camera)
- **USB 3.0 ports** (2x)
- **PC with GPU recommended** (for real-time processing)

### Software
- **Python 3.8 - 3.11**
- **Windows 10/11** or **Ubuntu 20.04+**

### Python Dependencies
```bash
# Core libraries
opencv-python>=4.8.0
numpy>=1.24.0
mediapipe>=0.10.0
pyrealsense2>=2.54.0

# GUI & visualization
tkinter (usually pre-installed)
pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Data processing
pandas>=2.0.0
scipy>=1.11.0
filterpy>=1.4.5
```

---

## 🚀 Installation

### Step 1: Clone/Download Repository
```bash
# Download the three Python files:
# - camera_calibration.py
# - dual_camera_tracker.py
# - data_analyzer.py
```

### Step 2: Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install opencv-python numpy mediapipe pyrealsense2 pillow matplotlib seaborn pandas scipy filterpy
```

### Step 3: Connect Cameras
1. Connect **RealSense D435i** to USB 3.0 port
2. Connect **Webcam/GoPro** to another USB port
3. Verify camera indices:
   ```bash
   # Test RealSense
   python -c "import pyrealsense2 as rs; print('RealSense OK')"
   
   # Test webcam (may be index 0, 1, or 2)
   python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'Try index 1 or 2')"
   ```

---

## 🎯 Quick Start Guide

### Workflow Overview
```
1. Calibrate Cameras → 2. Record Foot Movement → 3. Analyze Data
   (camera_calibration.py)   (dual_camera_tracker.py)   (data_analyzer.py)
```

### Step-by-Step

#### **1️⃣ Camera Calibration** (First Time Setup)

```bash
python camera_calibration.py
```

**Instructions:**
1. Click **"Generate Pattern"** → Print the generated PNG file
2. Click **"Start Cameras"**
3. Click **"Start Calibration"**
4. Show the printed pattern to **BOTH cameras**
5. When **"DETECTED"** appears, press **SPACE** to capture (15-20 times)
6. Vary the angle and distance of the pattern
7. Press **C** to calculate calibration
8. Calibration saved to `camera_calibration.json`

**Tips:**
- Use **ChArUco** pattern for better accuracy
- Ensure good lighting
- Cover the entire camera field of view

---

#### **2️⃣ Record Foot Movement**

```bash
python dual_camera_tracker.py
```

**Instructions:**
1. Select **Webcam Index** (0, 1, or 2) - test which works
2. Choose **Left** or **Right** foot
3. Enable **"📹 Save Video with CSV"** checkbox
4. Click **"Start Cameras"** → Wait for feed to appear
5. Click **"Start Recording"**
6. Perform foot movement (walking, ankle exercises, etc.)
7. Click **"Stop Recording"** when done
8. Files saved:
   - `foot_data_YYYYMMDD_HHMMSS.csv`
   - `foot_video_YYYYMMDD_HHMMSS.mp4`

**Tips:**
- Ensure foot is visible to **BOTH cameras**
- Maintain good lighting
- Avoid occlusion (blocking the view)
- Check the 3D plot for real-time tracking quality

---

#### **3️⃣ Analyze Data**

```bash
python data_analyzer.py
```

**Instructions:**
1. Click **"Open CSV"** → Select your `foot_data_*.csv` file
2. Click **"Open Video"** → Select corresponding `foot_video_*.mp4`
3. (Optional) Click **"Load Reference Trajectory"** → Select `.txt` file
4. Use **Play/Pause** controls to review synchronized data
5. Click plot buttons to analyze:
   - **Position 3D** → 3D trajectory
   - **Position vs Time** → X/Y/Z over time
   - **Ankle Velocity** → Speed analysis
   - **Angles Over Time** → Plantar/dorsiflexion angles
   - **Trajectory Comparison** → Compare with reference
   - **Deviation Analysis** → RMSE calculation
6. Click **"Export Report"** to save analysis

**Advanced Features:**
- **Gait Cycle Validator**: Automatically detects & validates cycles
- **Filter Data Button**: Nudges trajectory toward reference (advanced)
- **Frame Slider**: Scrub through video frame-by-frame

---

## 📂 Program Details

### 1. Camera Calibration Tool
**File:** `camera_calibration.py`

#### Purpose
Calibrates the stereo camera system to enable accurate 3D triangulation.

#### Key Functions
```python
def detect_checkerboard()  # Detects checkerboard pattern
def detect_charuco()       # Detects ChArUco pattern (better)
def perform_calibration()  # Calculates camera matrices
def save_calibration()     # Exports to JSON
```

#### Output File
```json
{
  "camera_matrix_1": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs_1": [k1, k2, p1, p2, k3],
  "camera_matrix_2": [...],
  "R": [[rotation matrix]],
  "T": [[translation vector]],
  "baseline": 0.25,  // meters
  "timestamp": "2025-01-27 10:30:00"
}
```

#### Keyboard Shortcuts
- **SPACE** → Capture current frame
- **C** → Calculate calibration
- **ESC** → Cancel calibration

---

### 2. Dual Camera Foot Tracker
**File:** `dual_camera_tracker.py`

#### Purpose
Records 3D foot movement using MediaPipe pose estimation and stereo triangulation.

#### Key Functions
```python
def triangulate_point()              # 3D reconstruction
def calculate_plantar_dorsiflexion() # Angle calculation
def calculate_inversion_eversion()   # Angle calculation
def detect_gait_cycles()             # Gait analysis
```

#### Data Flow
```
Camera Frames → MediaPipe → 2D Keypoints → Triangulation → 3D Position
                                                              ↓
                                                          CSV + Video
```

#### CSV Output Columns
| Column | Description | Unit |
|--------|-------------|------|
| `timestamp` | Time since recording start | seconds |
| `frame` | Frame number | - |
| `ankle_x_3d` | Ankle X position | meters |
| `ankle_y_3d` | Ankle Y (height) | meters |
| `ankle_z_3d` | Ankle Z (depth) | meters |
| `heel_x_3d`, `toe_x_3d`, ... | Heel & toe positions | meters |
| `ankle_velocity_x/y/z` | Velocity components | m/s |
| `ankle_speed` | Total speed | m/s |
| `plantar_dorsi_angle` | P/D angle | degrees |
| `inversion_eversion_angle` | I/E angle | degrees |
| `triangulation_quality` | Quality metric (0-1) | - |

---

### 3. Data Analyzer
**File:** `data_analyzer.py`

#### Purpose
Post-process recorded data with advanced filtering and visualization.

#### Key Classes
```python
class DataFilter:
    - remove_outliers()      # Z-score based outlier removal
    - kalman_filter_1d()     # Kalman smoothing
    - butterworth_filter()   # Low-pass filtering
    - apply_full_pipeline()  # Complete filtering chain
```

```python
class FootDataAnalyzer:
    - detect_gait_cycles()   # Automatic gait detection
    - calculate_rmse()       # Error vs reference
    - plot_trajectory_comparison()  # Visual comparison
    - export_report()        # Text report generation
```

#### Filtering Pipeline
```
Raw Data → Outlier Removal → Interpolation → Kalman Filter → Butterworth Filter → Clean Data
           (Z-score > 3)      (NaN filling)   (R=5, Q=0.1)    (cutoff=5Hz)
```

#### Reference Trajectory Format
Create a `.txt` file with X, Y coordinates (in mm):
```
0, 0
50, 10
100, 25
150, 30
200, 25
250, 10
300, 0
```

#### RMSE Calculation
The analyzer computes **Root Mean Square Error** between actual and reference:
```
RMSE_total = sqrt(mean((X_actual - X_ref)² + (Y_actual - Y_ref)²))
```

**Interpretation:**
- **< 10 mm**: Excellent match
- **10-30 mm**: Good match
- **30-50 mm**: Fair match
- **> 50 mm**: Poor match

---

## 📊 Data Output

### Directory Structure
```
project_folder/
├── camera_calibration.json       # Calibration parameters
├── foot_data_20250127_103045.csv # Recorded data
├── foot_video_20250127_103045.mp4 # Synchronized video
├── checkerboard_6x9_25mm.png     # Generated pattern
└── analysis_report.txt           # Exported analysis
```

### CSV Data Example
```csv
timestamp,frame,ankle_x_3d,ankle_y_3d,ankle_z_3d,...
0.000,0,0.1234,0.0567,0.8901,...
0.033,1,0.1245,0.0580,0.8910,...
0.067,2,0.1250,0.0595,0.8905,...
```

### Analysis Report Example
```
==========================================================
FOOT MOVEMENT ANALYSIS REPORT WITH GAIT VALIDATION
==========================================================

CSV File: foot_data_20250127_103045.csv
Generated: 2025-01-27 10:35:20
Data Type: Dual Camera

SUMMARY STATISTICS
------------------------------------------------------------------
Total Frames: 1523
Duration: 50.77 seconds
Sampling Rate: 30.0 Hz

GAIT CYCLE ANALYSIS
------------------------------------------------------------------
Total Gait Cycles Detected: 8
Average Cycle Duration: 1.243 ± 0.089 s
Average Stride Length: 0.4521 ± 0.0234 m
Average P/D ROM: 28.45 ± 3.21°
Average I/E ROM: 12.34 ± 2.10°

TRAJECTORY COMPARISON (RMSE)
------------------------------------------------------------------
Root Mean Square Error (Shape-based):
  Total RMSE: 12.345 mm
  X RMSE: 8.234 mm
  Y RMSE: 9.123 mm

Interpretation:
  Good trajectory match (RMSE < 30mm)
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Camera Not Detected**
**Problem:** `Cannot open webcam at index 0`

**Solution:**
```bash
# Test different indices
python -c "import cv2; [print(f'Index {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
```
Change the **Webcam Index** in the GUI to the working index.

---

#### 2. **Poor Tracking Quality**
**Symptoms:** Jittery 3D trajectory, low visibility warnings

**Solutions:**
- ✅ Improve lighting (avoid shadows)
- ✅ Wear contrasting colored socks/shoes
- ✅ Ensure foot is visible to BOTH cameras
- ✅ Reduce distance to cameras (<2 meters)
- ✅ Recalibrate cameras

---

#### 3. **Calibration Fails**
**Error:** `Calibration failed: insufficient data`

**Solutions:**
- ✅ Capture **15+ frames** from different angles
- ✅ Ensure pattern is **flat** (not bent)
- ✅ Cover entire field of view
- ✅ Use **ChArUco** pattern instead of checkerboard
- ✅ Check that pattern is detected in **BOTH cameras**

---

#### 4. **Data Shows Zero Values**
**Problem:** CSV contains many zeros or NaNs

**Solutions:**
- ✅ Check calibration file exists (`camera_calibration.json`)
- ✅ Verify foot landmarks are detected (green markers on video)
- ✅ Ensure cameras are properly calibrated
- ✅ Use filtering in data analyzer to clean data

---

#### 5. **Video and CSV Not Synchronized**
**Problem:** Frame count mismatch

**Solution:**
- Always record with **"Save Video with CSV"** enabled
- If video is shorter, use frame slider range limit
- Re-record if synchronization is critical

---

## 📐 Technical Details

### Coordinate System
```
     Y (height)
     ↑
     |
     |
     └──────→ X (lateral)
    /
   /
  ↙ Z (depth)

RealSense Camera Frame:
- X: Left (+) / Right (-)
- Y: Up (+) / Down (-)
- Z: Away from camera (+)
```

### Angle Definitions

#### Plantar/Dorsiflexion
```
        Dorsiflexion (+)
             ↑
             |
    Heel ────┼──── Ankle
             |
             ↓
      Plantarflexion (-)
```

#### Inversion/Eversion
```
        Eversion (+)
             ↑
             |
    Toe ─────┼──── Heel
             |
             ↓
       Inversion (-)
```

### Filtering Parameters

#### Kalman Filter
- **R (Measurement Noise)**: 1-10 (lower = smoother, higher = more responsive)
- **Q (Process Noise)**: 0.01-0.5 (lower = trust model more)
- **Default**: R=5, Q=0.1 for general motion

#### Butterworth Filter
- **Cutoff Frequency**: 1-8 Hz
  - Walking: 2-3 Hz
  - Running: 4-5 Hz
  - Ankle exercises: 3-5 Hz
- **Order**: 4 (standard)

---

## 📚 Citations

### MediaPipe
```bibtex
@article{mediapipe2019,
  title={MediaPipe: A Framework for Building Perception Pipelines},
  author={Lugaresi, Camillo and Tang, Jiuqiang and Nash, Hadon and others},
  journal={arXiv preprint arXiv:1906.08172},
  year={2019}
}
```

### Kalman Filtering
```bibtex
@book{welch1995kalman,
  title={An introduction to the Kalman filter},
  author={Welch, Greg and Bishop, Gary},
  year={1995},
  publisher={University of North Carolina at Chapel Hill}
}
```

### OpenCV Calibration
```bibtex
@article{zhang2000flexible,
  title={A flexible new technique for camera calibration},
  author={Zhang, Zhengyou},
  journal={IEEE Transactions on pattern analysis and machine intelligence},
  volume={22},
  number={11},
  pages={1330--1334},
  year={2000}
}
```

---

## 📞 Support & Contact

### Issues?
1. Check [Troubleshooting](#troubleshooting) section
2. Verify hardware connections
3. Review console output for error messages
4. Ensure all dependencies are installed

### Feature Requests
This is a research tool. Customize as needed:
- Modify filtering parameters in `DataFilter` class
- Add new plots in `FootDataAnalyzer`
- Adjust MediaPipe confidence thresholds

---

## 📄 License

This software is provided for **research and educational purposes**.

**Disclaimer:** Not intended for medical diagnosis. Consult professionals for clinical applications.

---

## 🙏 Acknowledgments

- **Intel RealSense** team for depth sensing SDK
- **Google MediaPipe** team for pose estimation
- **OpenCV** community for computer vision tools
- **FilterPy** developers for Kalman filtering

---

**Version:** 1.0  
**Last Updated:** January 27, 2025  
**Author:** [Your Name/Institution]

---

## 🎓 Academic Use

If you use this system in your research, please cite:

```bibtex
@software{dual_camera_foot_tracker_2025,
  title={Dual Camera Foot Tracking System with Advanced Filtering},
  author={[Your Name]},
  year={2025},
  version={1.0},
  note={3D gait analysis using RealSense D435i and MediaPipe}
}
```

---

**Happy Tracking! 🚶‍♂️📊**