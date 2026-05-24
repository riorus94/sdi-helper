***

# Software Requirements Specification (SRS)

## Vehicle Proportion Analysis & SDI System

**Version:** 2  
**Status:** Final  
**Audience:** Product, Engineering, CV, Domain Experts

***

## 1. Introductionx

### 1.1 Purpose

Dokumen ini mendefinisikan kebutuhan fungsional dan non‑fungsional untuk **Vehicle Proportion Analysis & SDI System**, sebuah aplikasi analisis proporsi kendaraan berbasis citra yang menghasilkan **dimensi geometrik**, **rasio desain**, dan **Styling / Design Index (SDI)** secara deterministik dan dapat dijelaskan.

Dokumen ini dimaksudkan sebagai:

*   Kontrak fungsional sistem
*   Dasar desain teknis (TSD)
*   Referensi persetujuan implementasi

***

### 1.2 Scope

Sistem memungkinkan pengguna untuk:

*   Mengelola identitas kendaraan
*   Mengunggah citra kendaraan dari beberapa sudut pandang
*   Mengekstraksi keypoint geometrik
*   Menghitung dan menyesuaikan dimensi kendaraan (mm)
*   Mengonversi dimensi menjadi rasio proporsi
*   Melakukan seleksi tipe kendaraan terdekat
*   Menghitung dan menampilkan SDI
*   Menghasilkan dan menyimpan laporan PDF

Sistem ini **bukan** sistem prediksi berbasis machine learning end‑to‑end, melainkan **decision support system berbasis geometri dan domain rules**.

***

### 1.3 Definitions and Terminology

| Term      | Definition                               |
| --------- | ---------------------------------------- |
| CV        | Computer Vision                          |
| Keypoint  | Titik referensi geometrik pada kendaraan |
| Dimension | Nilai jarak geometrik dalam milimeter    |
| Ratio     | Perbandingan dua dimensi tanpa satuan    |
| SDI       | Styling / Design Index                   |
| Overlay   | Visualisasi garis dimensi pada citra     |
| View      | Sudut pandang citra: side, front, rear   |

***

## 2. Overall Description

### 2.1 Product Perspective

Sistem terdiri dari:

*   Frontend web application
*   Backend API
*   CV inference service
*   Domain logic (geometry, ratios, SDI)

Domain logic bersifat **deterministic** dan **independen dari model CV**.

***

### 2.2 User Classes

*   **Standard User**: Pengguna yang melakukan analisis kendaraan dan menghasilkan laporan
*   **Administrator (opsional)**: Mengelola konfigurasi rasio referensi, bobot SDI, dan data tipe kendaraan

***

### 2.3 Operating Environment

*   Browser modern (Chrome, Edge, Firefox)
*   Backend service (HTTP API)
*   CV inference service (GPU‑capable)
*   Database relasional

***

## 3. System Features and Functional Requirements

### 3.1 Authentication

*   **FR‑01**: Sistem harus menyediakan autentikasi menggunakan username dan password
*   **FR‑02**: Sistem harus membatasi seluruh fitur analisis hanya untuk user terautentikasi

***

### 3.2 Vehicle Identity Management

*   **FR‑03**: User harus dapat membuat entitas kendaraan dengan properti:
    *   Alias kendaraan (mandatory)
    *   Manufaktur (optional)
    *   Tipe/model (optional)
    *   Warna (optional)

***

### 3.3 Image Input (Multi‑View)

*   **FR‑04**: Sistem harus menerima citra kendaraan dengan tipe view:
    *   Side (tampak samping)
    *   Front (tampak depan)
    *   Rear (tampak belakang)
*   **FR‑05**: Sistem harus menyimpan citra asli tanpa modifikasi

***

### 3.4 Keypoint Extraction

*   **FR‑06**: Sistem harus mengekstraksi keypoint:
    *   Side view: A, J, D, E, F1–F4, G, H, I
    *   Front view: M, N, O
    *   Rear view: P, Q
*   **FR‑07**: Setiap keypoint harus disimpan beserta view dan koordinat

***

### 3.5 Dimension Calculation

*   **FR‑08**: Sistem harus menghitung **20 dimensi geometrik utama** (mm)
*   **FR‑09**: Sistem harus menampilkan overlay garis dimensi pada citra

***

### 3.6 Interactive Dimension Adjustment

*   **FR‑10**: Sistem menampilkan 20 nilai dimensi dalam input (mm)
*   **FR‑11**: User dapat mengubah nilai dimensi secara manual
*   **FR‑12**: Perubahan harus memicu:
    *   Perhitungan ulang geometri
    *   Update overlay otomatis

***

### 3.7 Ratio Conversion

*   **FR‑13**: Sistem mengonversi 20 dimensi menjadi **12 rasio desain**
*   **FR‑14**: Sistem melakukan normalisasi rasio berdasarkan batas konfigurasi

***

### 3.8 Vehicle Type Selection

*   **FR‑15**: Sistem membandingkan rasio dengan database referensi
*   **FR‑16**: Sistem menghasilkan daftar tipe kendaraan berdasarkan kemiripan

***

### 3.9 SDI Calculation

*   **FR‑17**: Sistem menghitung nilai SDI dari rasio ter-normalisasi
*   **FR‑18**: Sistem menampilkan:
    *   Nilai SDI numerik
    *   Interpretasi karakter desain

***

### 3.10 Reporting and Persistence

*   **FR‑19**: Sistem menghasilkan laporan PDF berisi:
    *   Identitas kendaraan
    *   Citra + overlay
    *   Dimensi & rasio
    *   Seleksi tipe kendaraan
    *   Nilai SDI & interpretasi
*   **FR‑20**: Sistem menyimpan hasil analisis dan laporan

***

## 4. Non-Functional Requirements

### 4.1 Accuracy & Consistency

*   **NFR‑01**: Output harus konsisten untuk input yang sama

***

### 4.2 Explainability

*   **NFR‑02**: Nilai SDI harus dapat ditelusuri ke rasio, dimensi, dan keypoint

***

### 4.3 Usability

*   **NFR‑03**: Relasi dimensi dan overlay harus mudah dipahami

***

### 4.4 Maintainability

*   **NFR‑04**: Perubahan model CV tidak memengaruhi domain logic

***

## 5. Constraints and Assumptions

### 5.1 Constraints

*   Menggunakan citra 2D
*   Tidak memerlukan kalibrasi kamera wajib

***

### 5.2 Assumptions

*   Citra diambil dengan sudut pandang yang layak
*   Roda terlihat sebagai referensi skala

***

## 6. Out of Scope

*   Prediksi desain berbasis machine learning end‑to‑end
*   Analisis performa mesin atau aerodinamika
*   Rekonstruksi 3D

***

## 7. Acceptance Criteria

Sistem dianggap memenuhi SRS apabila:

*   Flow dari input citra hingga PDF berjalan utuh
*   User dapat mengoreksi dimensi tanpa inkonsistensi visual
*   SDI bersifat reproducible dan explainable

***

## 8. Conclusion

SRS ini mendefinisikan sistem sebagai **tool analisis desain kendaraan yang berbasis geometri, deterministic, dan explainable**.

Dokumen ini menjadi **acuan final** bagi desain teknis, implementasi, dan validasi sistem.

***
