# Identifikasi Pemalsuan Bubuk Cabai Merah Menggunakan Citra Digital Berbasis EfficientNetV2 dengan Visualisasi Explainable AI

Repository ini berisi kode untuk skripsi di atas: klasifikasi citra bubuk cabai
merah murni vs campuran menggunakan EfficientNetV2S (transfer learning +
fine-tuning), dilengkapi visualisasi Grad-CAM untuk interpretability model.

## Struktur Repo

```
.
├── split_data.py       # Analisis dataset & stratified split (80:10:10)
├── efficientnetv2.py   # Training 2-stage (transfer learning + fine-tuning) & evaluasi
├── xai_gradcam.py       # Visualisasi Grad-CAM untuk interpretability
├── requirements.txt
├── .gitignore
└── docs/
    └── figures/         # Beberapa contoh hasil (opsional, untuk ilustrasi di laporan)
```

## Yang TIDAK ada di repo ini (dan alasannya)

| Item | Alasan |
|---|---|
| Dataset mentah citra | Ukuran besar, bukan tempatnya di repo kode |
| Folder `data_split/` | Hasil generate otomatis dari `split_data.py`, tidak perlu di-commit |
| File model `.keras` | Kemungkinan >50-100MB, melebihi batas nyaman GitHub tanpa Git LFS |
| Seluruh output Grad-CAM (PNG/ZIP per kelas) | Ribuan file, hasil generate dari `xai_gradcam.py` |

Kalau perlu membagikan dataset tambahan, upload ke Kaggle Datasets / Google Drive /
Hugging Face, lalu cantumkan link-nya di bagian "Dataset" di bawah.

## Cara Menjalankan (Reproduksi Penuh)

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > **Catatan:** Versi pustaka pada `requirements.txt` merupakan estimasi umum,
   > bukan hasil `pip freeze` dari environment Kaggle yang sesungguhnya digunakan
   > selama penelitian. Untuk menjamin reproduksibilitas penuh, disarankan
   > menjalankan `pip freeze > requirements.txt` pada environment Kaggle yang
   > telah terverifikasi berjalan dengan baik, kemudian menggantikan berkas ini
   > dengan hasil tersebut.

2. **Siapkan dataset**
   Taruh dataset di folder `dataset/` dengan struktur:
   ```
   dataset/
   ├── KT00/
   ├── KTGM_5/
   ├── KTGM_10/
   ...
   ```
   (isi tabel penjelasan kode kelas di bawah)

3. **Preprocessing & split data**
   ```bash
   export DATASET_DIR=./dataset
   export SPLIT_DIR=./data_split
   python split_data.py
   ```
   Menghasilkan `data_split/{train,val,test}/<kelas>/` dan grafik distribusi di `figures/`.

4. **Training model**
   ```bash
   export TRAIN_DIR=./data_split/train
   export VAL_DIR=./data_split/val
   export TEST_DIR=./data_split/test
   export OUTPUT_ROOT=./outputs
   python efficientnetv2.py
   ```
   Menghasilkan model terlatih (`best_model_finetuned.keras`), metrik evaluasi,
   dan confusion matrix di `outputs/<nama_eksperimen>_<timestamp>/`.

5. **Visualisasi Grad-CAM (XAI)**
   ```bash
   export MODEL_PATH=./outputs/<nama_eksperimen>_<timestamp>/best_model_finetuned.keras
   export TEST_DIR=./data_split/test
   export GRADCAM_OUTPUT_DIR=./gradcam_output
   python xai_gradcam.py
   ```
   Menghasilkan heatmap Grad-CAM per kelas (dipisah benar/salah prediksi),
   dikemas sebagai ZIP per kelas.

## Penjelasan Kode Kelas

Dataset terdiri dari 16 kelas: 1 kelas murni dan 15 kelas hasil pemalsuan
menggunakan 5 jenis bahan pemalsu, masing-masing pada 3 tingkat konsentrasi
(5%, 10%, 15%).

| Kode | Arti |
|---|---|
| `KT00` | Bubuk Khammam Teja murni (tanpa pemalsuan) |
| `KTUT_5` / `KTUT_10` / `KTUT_15` | Bubuk Khammam Teja dengan pemalsu United sebesar 5% / 10% / 15% |
| `KTGM_5` / `KTGM_10` / `KTGM_15` | Bubuk Khammam Teja dengan pemalsu Guntur Mix sebesar 5% / 10% / 15% |
| `KTWB_5` / `KTWB_10` / `KTWB_15` | Bubuk Khammam Teja dengan pemalsu Wheat Bran sebesar 5% / 10% / 15% |
| `KTWS_5` / `KTWS_10` / `KTWS_15` | Bubuk Khammam Teja dengan pemalsu Wood Saw (serbuk gergaji) sebesar 5% / 10% / 15% |
| `KTRB_5` / `KTRB_10` / `KTRB_15` | Bubuk Khammam Teja dengan pemalsu Rice Hull (sekam padi) sebesar 5% / 10% / 15% |

## Dataset

- Dataset: [Mendeley Data - ppy7vg8h7z](https://data.mendeley.com/datasets/ppy7vg8h7z/2)

Model terlatih (`.keras`) tidak dibagikan lewat repo ini karena ukurannya
melebihi batas nyaman GitHub. Model dapat dilatih ulang dari nol mengikuti
langkah "Cara Menjalankan" di atas.

## Catatan Metodologis

- Eksperimen training ini (`expA`) sengaja tanpa Dropout, tanpa augmentasi data,
  dan tanpa label smoothing -- bagian dari perbandingan eksperimen di skripsi.
- Grad-CAM mengasumsikan layer konvolusi terakhir dalam urutan `backbone.layers`
  sama dengan layer yang secara aktual dieksekusi terakhir pada forward pass.
  Untuk arsitektur dengan skip connection (EfficientNetV2S), ini asumsi yang
  wajar tapi bukan jaminan mutlak -- lihat komentar di `xai_gradcam.py`.
