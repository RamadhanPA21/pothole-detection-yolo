# Deteksi Jalan Berlubang (Pothole Detection) menggunakan YOLOv8 dan YOLOv11

Proyek Tugas Akhir ini berfokus pada implementasi dan studi komparatif algoritma Deep Learning berbasis Computer Vision untuk mendeteksi kerusakan jalan (jalan berlubang) secara real-time. Proyek ini membandingkan arsitektur YOLOv8 dengan model generasi terbaru yaitu YOLOv11 guna menemukan model paling optimal untuk sistem pemantauan infrastruktur jalan cerdas.

## 🚀 Fitur Utama
* **Deteksi Real-Time:** Mampu memproses input video atau kamera dasbor (dashcam) secara langsung dengan visualisasi kotak pembatas (bounding box).
* **Studi Komparatif:** Analisis performa komparatif antara arsitektur YOLOv8 dan YOLOv11 berdasarkan tingkat akurasi serta kecepatan komputasi.
* **Integrasi Streamlit:** Antarmuka web interaktif yang memudahkan pengguna untuk mengunggah file video/gambar dan melihat hasil prediksi deteksi lubang secara instan.

## 🛠️ Tech Stack
* **Bahasa Pemrograman:** Python 3.x
* **Framework AI:** Ultralytics YOLO (v8 & v11), PyTorch
* **Pengolahan Citra:** OpenCV
* **Antarmuka Web:** Streamlit

## 📊 Kesimpulan Hasil Analisis (Benchmarking)
Berdasarkan serangkaian proses pelatihan (*training*) dan pengujian (*testing*) yang dilakukan pada proyek Tugas Akhir ini, diperoleh beberapa kesimpulan utama mengenai performa kedua model:
* **Akurasi Deteksi:** Model **YOLOv11** menunjukkan performa akurasi yang lebih unggul dalam mengenali objek jalan berlubang, terutama pada kondisi pencahayaan yang minim (*low light*) dan objek berukuran kecil di kejauhan.
* **Kecepatan Inferensi:** Kedua model mampu berjalan secara *real-time*, namun **YOLOv11** menawarkan efisiensi parameter yang lebih baik sehingga menghasilkan komputasi yang lebih ringan tanpa mengorbankan akurasi.

## 💻 Panduan Menjalankan Proyek
1. Clone repositori ini ke komputer lokal Anda:
   ```bash
   git clone https://github.com
   ```
2. Jalankan aplikasi web interaktif melalui terminal:
   ```bash
   streamlit run src/streamlit_app.py
   ```
