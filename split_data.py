"""
Preprocessing & Stratified Split Data

Identifikasi Pemalsuan Bubuk Cabai Merah Menggunakan Citra Digital
Berbasis EfficientNetV2 dengan Visualisasi Explainable AI

Pipeline lengkap ada 3 tahap:
1. Preprocessing & split data       
2. Training & evaluasi model 
3. Visualisasi Grad-CAM (XAI)  

Output: folder data_split/{train,val,test}/<kelas>/ berisi salinan
gambar sesuai pembagian stratified 80:10:10. Folder ini TIDAK perlu
di-commit ke repo, cukup jalankan ulang script ini untuk generate.
"""

import os
import random
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------
# Konfigurasi
# ------------------------------------------------------------------

@dataclass
class Config:
    # Ganti sesuai lokasi dataset di environment-mu (Kaggle/Colab/lokal).
    dataset_dir: str = os.environ.get("DATASET_DIR", "./dataset")
    split_dir: str = os.environ.get("SPLIT_DIR", "./data_split")
    figures_dir: str = os.environ.get("FIGURES_DIR", "./figures")

    random_seed: int = 42
    val_test_size: float = 0.2   # 20% dipisah untuk val+test
    test_ratio_of_temp: float = 0.5  # dari 20% itu, setengahnya jadi test (-> 10% total)


CLASS_NAMES = [
    "KT00",
    "KTGM_5", "KTGM_10", "KTGM_15",
    "KTRB_5", "KTRB_10", "KTRB_15",
    "KTUT_5", "KTUT_10", "KTUT_15",
    "KTWB_5", "KTWB_10", "KTWB_15",
    "KTWS_5", "KTWS_10", "KTWS_15",
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# ------------------------------------------------------------------
# Analisis dataset
# ------------------------------------------------------------------

def analyze_dataset(dataset_dir, class_names, figures_dir):
    """Hitung jumlah citra per kelas dan simpan grafik distribusinya."""
    rows = []
    for class_name in class_names:
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_path):
            print(f"WARNING: folder kelas tidak ditemukan: {class_name}")
            continue
        count = len([f for f in os.listdir(class_path) if f.lower().endswith(IMAGE_EXTENSIONS)])
        rows.append({"Kelas": class_name, "Jumlah Citra": count})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"Total citra: {df['Jumlah Citra'].sum()}")

    os.makedirs(figures_dir, exist_ok=True)
    plt.figure(figsize=(14, 6))
    bars = plt.bar(df["Kelas"], df["Jumlah Citra"], color="steelblue", edgecolor="black")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Citra")
    plt.title(f"Distribusi Dataset per Kelas (Total: {df['Jumlah Citra'].sum():,} Citra)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/distribusi_dataset.png", dpi=300, bbox_inches="tight")
    plt.close()

    return df


# ------------------------------------------------------------------
# Stratified split
# ------------------------------------------------------------------

def list_files_and_labels(dataset_dir, class_names):
    file_paths, labels = [], []
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_path):
            continue
        for img in os.listdir(class_path):
            if img.lower().endswith(IMAGE_EXTENSIONS):
                file_paths.append(os.path.join(class_path, img))
                labels.append(idx)
    return np.array(file_paths), np.array(labels)


def stratified_split(files, labels, cfg: Config):
    """80% train, 10% val, 10% test, stratified per kelas."""
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files, labels,
        test_size=cfg.val_test_size,
        stratify=labels,
        random_state=cfg.random_seed,
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=cfg.test_ratio_of_temp,
        stratify=temp_labels,
        random_state=cfg.random_seed,
    )
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


def verify_split(splits, class_names):
    """Cetak tabel jumlah data per kelas di tiap split, untuk memastikan proporsi stratified benar."""
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = splits
    rows = []
    for idx, class_name in enumerate(class_names):
        t = int(np.sum(train_labels == idx))
        v = int(np.sum(val_labels == idx))
        te = int(np.sum(test_labels == idx))
        rows.append({"Kelas": class_name, "Train": t, "Val": v, "Test": te, "Total": t + v + te})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def plot_split_distribution(split_df, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, split_name, color in zip(axes, ["Train", "Val", "Test"], ["#2ecc71", "#3498db", "#e74c3c"]):
        ax.bar(split_df["Kelas"], split_df[split_name], color=color, edgecolor="black", alpha=0.85)
        ax.set_title(f"{split_name} set ({split_df[split_name].sum()} citra)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/distribusi_split.png", dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------
# Salin file ke struktur folder split
# ------------------------------------------------------------------

def copy_split_to_disk(files, labels, target_dir, class_names):
    for class_name in class_names:
        os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)
    for file_path, label in zip(files, labels):
        class_name = class_names[label]
        target_path = os.path.join(target_dir, class_name, os.path.basename(file_path))
        shutil.copy2(file_path, target_path)
    print(f"{len(files)} file disalin ke {target_dir}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg = Config()
    set_seed(cfg.random_seed)

    if not os.path.exists(cfg.dataset_dir):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {cfg.dataset_dir}")

    analyze_dataset(cfg.dataset_dir, CLASS_NAMES, cfg.figures_dir)

    files, labels = list_files_and_labels(cfg.dataset_dir, CLASS_NAMES)
    print(f"Total file: {len(files)} | Total kelas: {len(np.unique(labels))}")

    splits = stratified_split(files, labels, cfg)
    split_df = verify_split(splits, CLASS_NAMES)
    plot_split_distribution(split_df, cfg.figures_dir)

    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = splits
    copy_split_to_disk(train_files, train_labels, os.path.join(cfg.split_dir, "train"), CLASS_NAMES)
    copy_split_to_disk(val_files, val_labels, os.path.join(cfg.split_dir, "val"), CLASS_NAMES)
    copy_split_to_disk(test_files, test_labels, os.path.join(cfg.split_dir, "test"), CLASS_NAMES)

    print(f"\nSelesai. Struktur data split ada di: {cfg.split_dir}")


if __name__ == "__main__":
    main()