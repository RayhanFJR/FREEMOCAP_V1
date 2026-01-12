# FreeMoCap - Motion Capture System

Sistem motion capture menggunakan webcam dan Intel RealSense D435i dengan dukungan depth camera.

## Fitur

- Kalibrasi kamera untuk webcam dan RealSense D435i
- Capture video dari kedua kamera secara simultan
- Pemanfaatan depth data dari RealSense untuk akurasi lebih baik
- Pose estimation menggunakan MediaPipe
- UI untuk kontrol dan visualisasi

## Instalasi

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Pastikan RealSense SDK sudah terinstall di sistem Anda

## Penggunaan

Jalankan aplikasi utama:
```bash
python main.py
```

## Struktur Proyek

```
FreemocapV1/
├── main.py                 # Entry point aplikasi
├── src/                    # Source code
│   ├── camera/            # Modul capture video
│   │   ├── __init__.py
│   │   └── camera_capture.py
│   ├── calibration/       # Modul kalibrasi kamera
│   │   ├── __init__.py
│   │   ├── camera_calibration.py
│   │   └── stereo_calibration.py
│   ├── pose/              # Modul pose estimation
│   │   ├── __init__.py
│   │   ├── pose_estimation.py
│   │   └── triangulation.py
│   ├── gui/               # UI components
│   │   ├── __init__.py
│   │   └── ui_main.py
│   ├── visualization/     # Visualisasi data
│   │   ├── __init__.py
│   │   ├── visualize_gui.py
│   │   └── visualize_pose.py
│   └── utils/             # Utility functions
│       ├── __init__.py
│       └── utils.py
└── requirements.txt
```

