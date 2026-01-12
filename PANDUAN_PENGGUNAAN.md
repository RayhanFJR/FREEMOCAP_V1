# Panduan Penggunaan FreeMoCap

## Persiapan

### 1. Install Dependencies

Buka terminal/command prompt dan jalankan:

```bash
pip install -r requirements.txt
```

Pastikan semua package terinstall dengan benar:
- opencv-python
- pyrealsense2
- mediapipe
- numpy
- PyQt5
- scipy
- matplotlib
- pandas

### 2. Persiapan Hardware

- **Webcam**: Pastikan webcam terhubung dan terdeteksi oleh sistem
- **RealSense D435i**: 
  - Pasang kabel USB 3.0 ke komputer
  - Pastikan Intel RealSense SDK sudah terinstall
  - Download dari: https://www.intelrealsense.com/sdk-2/
- **Checkerboard**: Siapkan checkerboard untuk kalibrasi (default: 7x5 kotak, ukuran 20mm per kotak)

## Menjalankan Aplikasi

Jalankan aplikasi dengan command:

```bash
python main.py
```

Atau:

```bash
python -m main
```

## Langkah-langkah Penggunaan

### TAHAP 1: Kalibrasi Kamera

Kalibrasi diperlukan untuk mendapatkan akurasi yang baik dalam motion capture.

#### 1.1 Buka Tab "Kalibrasi"

Setelah aplikasi terbuka, klik pada tab **"Kalibrasi"** di bagian atas.

#### 1.2 Atur Settings Kalibrasi

Di bagian "Settings Kalibrasi":
- **Checkerboard Columns**: Atur jumlah kolom checkerboard (default: 7)
- **Checkerboard Rows**: Atur jumlah baris checkerboard (default: 5)
- **Square Size (mm)**: Atur ukuran setiap kotak dalam milimeter (default: 20.0)

**PENTING**: Pastikan nilai ini sesuai dengan checkerboard fisik yang Anda gunakan!

#### 1.3 Start Capture

1. Klik tombol **"Start Capture"** untuk mulai streaming dari kamera
2. Pastikan kedua kamera (webcam dan RealSense) terdeteksi
3. Jika ada error, pastikan:
   - Webcam tidak digunakan aplikasi lain
   - RealSense terhubung dengan USB 3.0
   - Driver RealSense sudah terinstall

#### 1.4 Capture Gambar Kalibrasi

1. Tampilkan checkerboard di depan kamera
2. Pastikan checkerboard terlihat jelas di kedua kamera
3. Gerakkan checkerboard ke berbagai posisi dan sudut:
   - Posisi tengah, kiri, kanan
   - Sudut miring ke berbagai arah
   - Jarak dekat dan jauh
4. Setiap kali checkerboard terdeteksi dengan baik, klik **"Capture Image"**
5. Ulangi minimal **10 kali** (lebih banyak lebih baik, disarankan 15-20 gambar)
6. Status akan muncul di text box di bawah:
   - "Checkerboard ditemukan!" = berhasil
   - "Checkerboard tidak ditemukan" = coba lagi dengan posisi berbeda

**Tips**:
- Pastikan pencahayaan cukup
- Checkerboard harus datar (tidak melengkung)
- Hindari refleksi pada checkerboard
- Variasikan posisi dan sudut sebanyak mungkin

#### 1.5 Lakukan Kalibrasi

1. Setelah memiliki minimal 10 gambar, klik tombol **"Kalibrasi"**
2. Tunggu proses kalibrasi selesai (biasanya beberapa detik)
3. Status akan menampilkan hasil:
   - "Webcam kalibrasi selesai! Error: X.XXXX"
   - "RealSense kalibrasi selesai! Error: X.XXXX"
4. Error yang lebih kecil = hasil lebih baik (biasanya < 1.0)

#### 1.6 Simpan Hasil Kalibrasi

1. Klik tombol **"Save Calibration"**
2. Pilih lokasi penyimpanan
3. File akan disimpan sebagai:
   - `[nama_file]_webcam.json`
   - `[nama_file]_realsense.json`
4. Simpan dengan nama yang mudah diingat, misalnya: `calibration_2024_webcam.json`

#### 1.7 (Opsional) Load Kalibrasi Lama

Jika sudah pernah melakukan kalibrasi sebelumnya:
1. Klik **"Load Calibration"**
2. Pilih file kalibrasi yang ingin dimuat
3. Kalibrasi akan langsung digunakan

#### 1.8 Reset Kalibrasi

Jika ingin memulai kalibrasi dari awal:
- Klik **"Reset"** untuk menghapus semua gambar kalibrasi yang sudah di-capture

---

### TAHAP 2: Motion Capture

Setelah kalibrasi selesai, Anda bisa mulai melakukan motion capture.

#### 2.1 Buka Tab "Motion Capture"

Klik pada tab **"Motion Capture"** di bagian atas aplikasi.

#### 2.2 Start Capture

1. Klik tombol **"Start Capture"**
2. Anda akan melihat 3 tampilan:
   - **Webcam**: Video dari webcam
   - **RealSense Color**: Video color dari RealSense
   - **RealSense Depth**: Visualisasi depth map dari RealSense (warna menunjukkan jarak)

#### 2.3 Aktifkan Pose Estimation

1. Centang checkbox **"Enable Pose Estimation"**
2. Pose estimation akan mendeteksi pose tubuh manusia dalam frame
3. Skeleton akan ditampilkan di video RealSense Color
4. Landmark points (titik-titik tubuh) akan ditampilkan dengan garis-garis

**Tips untuk pose estimation yang baik**:
- Pastikan seluruh tubuh terlihat di frame
- Pencahayaan cukup
- Background tidak terlalu ramai
- Berdiri menghadap kamera
- Jarak optimal: 1-3 meter dari kamera

#### 2.4 Mulai Recording

1. Klik tombol **"Start Recording"** untuk mulai merekam data pose
2. Tombol akan berubah menjadi **"Stop Recording"**
3. Status akan menampilkan "Status: Recording..."
4. Gerakkan tubuh Anda - semua pose akan direkam
5. Data pose akan disimpan dalam memori

#### 2.5 Stop Recording

1. Klik **"Stop Recording"** untuk menghentikan perekaman
2. Dialog akan muncul untuk menyimpan file
3. Pilih lokasi dan nama file (format: JSON)
4. Data akan berisi:
   - Landmarks 2D (koordinat pixel)
   - Landmarks 3D (koordinat 3D menggunakan depth data)
   - Timestamp untuk setiap frame
   - Nama-nama landmark (nose, shoulder, elbow, dll)

#### 2.6 Stop Capture

Setelah selesai:
1. Klik **"Stop Capture"** untuk menghentikan streaming
2. Semua kamera akan dilepas

---

## Format Data Output

Data yang direkam disimpan dalam format JSON dengan struktur:

```json
[
  {
    "landmarks_2d": [[x1, y1], [x2, y2], ...],
    "landmarks_3d": [[x1, y1, z1], [x2, y2, z2], ...],
    "landmark_names": ["nose", "left_eye", ...],
    "timestamp": 123456.789
  },
  ...
]
```

Setiap frame berisi:
- **landmarks_2d**: 33 titik koordinat 2D (pixel)
- **landmarks_3d**: 33 titik koordinat 3D (meter, menggunakan depth)
- **landmark_names**: Nama setiap landmark
- **timestamp**: Waktu capture dalam detik

## Troubleshooting

### Kamera tidak terdeteksi
- Pastikan webcam tidak digunakan aplikasi lain
- Restart aplikasi
- Cek koneksi USB untuk RealSense

### RealSense tidak terdeteksi
- Pastikan menggunakan USB 3.0 port
- Install Intel RealSense SDK
- Cek di Device Manager apakah RealSense terdeteksi

### Checkerboard tidak terdeteksi
- Pastikan pencahayaan cukup
- Checkerboard harus datar
- Coba dengan ukuran checkerboard yang berbeda
- Pastikan semua kotak terlihat jelas

### Pose estimation tidak akurat
- Pastikan seluruh tubuh terlihat
- Perbaiki pencahayaan
- Jarak optimal 1-3 meter
- Background tidak terlalu ramai

### Aplikasi crash atau error
- Pastikan semua dependencies terinstall
- Cek versi Python (disarankan Python 3.8+)
- Pastikan RealSense SDK terinstall dengan benar

## Tips & Best Practices

1. **Kalibrasi berkala**: Lakukan kalibrasi ulang jika:
   - Kamera dipindahkan
   - Fokus kamera berubah
   - Hasil capture tidak akurat

2. **Pencahayaan**: 
   - Gunakan pencahayaan yang merata
   - Hindari bayangan yang terlalu kuat
   - Hindari backlight yang terlalu terang

3. **Posisi kamera**:
   - Letakkan kamera setinggi tubuh
   - RealSense dan webcam sebaiknya berdekatan
   - Pastikan tidak ada halangan di depan kamera

4. **Recording**:
   - Rekam dalam durasi pendek-pendek untuk menghindari file terlalu besar
   - Beri nama file yang deskriptif
   - Simpan di folder yang terorganisir

5. **Performance**:
   - Tutup aplikasi lain yang menggunakan kamera
   - Gunakan komputer dengan spesifikasi yang cukup
   - Jika lag, kurangi resolusi di kode

## Kontak & Support

Jika ada masalah atau pertanyaan, silakan cek:
- Dokumentasi OpenCV: https://docs.opencv.org/
- Dokumentasi MediaPipe: https://google.github.io/mediapipe/
- Dokumentasi RealSense: https://www.intelrealsense.com/sdk-2/

