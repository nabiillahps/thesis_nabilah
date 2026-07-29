"""
Identifikasi Pemalsuan Bubuk Cabai Merah Menggunakan Citra Digital
Berbasis EfficientNetV2 dengan Visualisasi Explainable AI

Pipeline lengkap ada 3 tahap:
1. Preprocessing & split data       
2. Training & evaluasi model 
3. Visualisasi Grad-CAM (XAI) 

Training & Evaluation - EfficientNetV2S
"""

import os
import sys
import json
import shutil
import time
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, TensorBoard

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


# ------------------------------------------------------------------
# Konfigurasi
# ------------------------------------------------------------------

@dataclass
class Config:
    # Path data. Ubah sesuai environment (Kaggle/Colab/lokal).
    train_dir: str = os.environ.get("TRAIN_DIR", "./data_split/train")
    val_dir: str = os.environ.get("VAL_DIR", "./data_split/val")
    test_dir: str = os.environ.get("TEST_DIR", "./data_split/test")
    output_root: str = os.environ.get("OUTPUT_ROOT", "./outputs")

    input_shape: tuple = (224, 224, 3)
    batch_size: int = 64

    learning_rate: float = 0.0006
    epochs_stage1: int = 70
    epochs_stage2: int = 35
    fine_tune_at: int = 180
    dense_units: int = 64
    l2_reg: float = 0.0005

    experiment_name: str = "expA_no_dropout_no_aug_no_smooth"


# ------------------------------------------------------------------
# Metrik custom
# ------------------------------------------------------------------

class F1Score(keras.metrics.Metric):
    """F1-score dihitung dari precision & recall bawaan Keras."""

    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision_metric = keras.metrics.Precision()
        self.recall_metric = keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred, sample_weight)
        self.recall_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision_metric.result()
        r = self.recall_metric.result()
        return 2 * ((p * r) / (p + r + keras.backend.epsilon()))

    def reset_state(self):
        self.precision_metric.reset_state()
        self.recall_metric.reset_state()


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------

class EpochLogger(keras.callbacks.Callback):
    """Cetak ringkasan metrik tiap epoch dalam satu baris tabel."""

    def __init__(self, stage_name):
        super().__init__()
        self.stage_name = stage_name
        self.start_time = None
        self.epoch_start = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n[{self.stage_name}] mulai training")
        header = (
            f"{'epoch':>5} | {'acc':>7} | {'loss':>7} | {'f1':>7} | "
            f"{'val_acc':>7} | {'val_loss':>7} | {'val_f1':>7} | time"
        )
        print(header)
        print("-" * len(header))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        dur = time.time() - self.epoch_start
        print(
            f"{epoch + 1:>5} | {logs.get('accuracy', 0):>7.4f} | "
            f"{logs.get('loss', 0):>7.4f} | {logs.get('f1_score', 0):>7.4f} | "
            f"{logs.get('val_accuracy', 0):>7.4f} | {logs.get('val_loss', 0):>7.4f} | "
            f"{logs.get('val_f1_score', 0):>7.4f} | {dur:.0f}s"
        )
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        dur = time.time() - self.start_time
        print(f"[{self.stage_name}] selesai, durasi {dur / 60:.1f} menit")


class BestRecallCheckpoint(keras.callbacks.Callback):
    """Simpan model saat val_recall terbaik (tie-break: val_loss terendah)."""

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best_recall = -1.0
        self.best_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_recall = logs.get("val_recall", 0)
        val_loss = logs.get("val_loss", float("inf"))

        improved = val_recall > self.best_recall or (
            val_recall == self.best_recall and val_loss < self.best_loss
        )
        if improved:
            self.best_recall = val_recall
            self.best_loss = val_loss
            self.model.save(self.filepath)
            print(
                f"  checkpoint disimpan (epoch {epoch + 1}): "
                f"val_recall={val_recall:.4f}, val_loss={val_loss:.4f}"
            )


def build_callbacks(model_dir, filename, csv_name, stage_name):
    return [
        BestRecallCheckpoint(filepath=f"{model_dir}/{filename}"),
        CSVLogger(f"{model_dir}/{csv_name}", append=False),
        TensorBoard(log_dir=f"{model_dir}/logs"),
        EpochLogger(stage_name=stage_name),
    ]


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def build_generators(cfg: Config):
    """Generator tanpa augmentasi (bagian dari desain eksperimen ini)."""
    datagen = ImageDataGenerator()

    train_gen = datagen.flow_from_directory(
        cfg.train_dir,
        target_size=cfg.input_shape[:2],
        batch_size=cfg.batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_gen = datagen.flow_from_directory(
        cfg.val_dir,
        target_size=cfg.input_shape[:2],
        batch_size=cfg.batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=42,
    )
    test_gen = datagen.flow_from_directory(
        cfg.test_dir,
        target_size=cfg.input_shape[:2],
        batch_size=cfg.batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=42,
    )
    return train_gen, val_gen, test_gen


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

def build_model(num_classes, cfg: Config):
    """
    Head klasifikasi di atas EfficientNetV2S (ImageNet pretrained).
    Tanpa Dropout, tanpa Label Smoothing -> sesuai desain eksperimen A.
    """
    base_model = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=cfg.input_shape,
        include_preprocessing=True,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=cfg.input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        cfg.dense_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(cfg.l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=keras.regularizers.l2(cfg.l2_reg),
    )(x)

    model = models.Model(inputs, outputs, name="EfficientNetV2S_ExpA")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            F1Score(name="f1_score"),
        ],
    )
    return model, base_model


# ------------------------------------------------------------------
# Analisis overfitting
# ------------------------------------------------------------------

def analyze_fit(history_s1, history_s2):
    train_acc = history_s1.history["accuracy"] + history_s2.history["accuracy"]
    val_acc = history_s1.history["val_accuracy"] + history_s2.history["val_accuracy"]
    train_loss = history_s1.history["loss"] + history_s2.history["loss"]
    val_loss = history_s1.history["val_loss"] + history_s2.history["val_loss"]

    last5_gap = (sum(train_acc[-5:]) - sum(val_acc[-5:])) / 5

    if train_acc[-1] < 0.70 and val_acc[-1] < 0.70:
        status = "UNDERFITTING"
    elif last5_gap > 0.10:
        status = "OVERFITTING BERAT"
    elif last5_gap > 0.05:
        status = "OVERFITTING RINGAN"
    elif train_acc[-1] >= 0.90 and val_acc[-1] >= 0.85 and last5_gap <= 0.05:
        status = "GOOD FIT"
    else:
        status = "PERLU EVALUASI LEBIH LANJUT"

    print(f"\nDiagnosis: {status} (gap 5-epoch terakhir: {last5_gap * 100:+.2f}%)")

    return {
        "status": status,
        "final_train_acc": float(train_acc[-1]),
        "final_val_acc": float(val_acc[-1]),
        "final_train_loss": float(train_loss[-1]),
        "final_val_loss": float(val_loss[-1]),
        "best_val_acc": float(max(val_acc)),
        "best_val_loss": float(min(val_loss)),
        "acc_gap_last5": float(last5_gap),
    }


# ------------------------------------------------------------------
# Visualisasi training history
# ------------------------------------------------------------------

METRICS_CONFIG = [
    ("accuracy", "val_accuracy", "Accuracy", "tab:blue"),
    ("loss", "val_loss", "Loss", "tab:red"),
    ("precision", "val_precision", "Precision", "tab:green"),
    ("recall", "val_recall", "Recall", "tab:orange"),
    ("f1_score", "val_f1_score", "F1-Score", "tab:purple"),
]


def plot_history_per_stage(history_s1, history_s2, save_dir):
    fig, axes = plt.subplots(2, 5, figsize=(30, 12))
    fig.suptitle("Training History per Stage", fontsize=15, fontweight="bold")

    stages = [(history_s1, "Stage 1: Transfer Learning"), (history_s2, "Stage 2: Fine-Tuning")]
    for col, (train_key, val_key, title, color) in enumerate(METRICS_CONFIG):
        for row, (history, stage_label) in enumerate(stages):
            ax = axes[row][col]
            train_vals = history.history.get(train_key, [])
            val_vals = history.history.get(val_key, [])
            x = range(1, len(train_vals) + 1)

            ax.plot(x, train_vals, label="Train", color=color, linewidth=2)
            ax.plot(x, val_vals, label="Val", color=color, linewidth=2, linestyle="--", alpha=0.8)

            best_idx = int(np.argmin(val_vals)) if "loss" in train_key else int(np.argmax(val_vals))
            ax.scatter(best_idx + 1, val_vals[best_idx], color="black", zorder=5, s=60)

            ax.set_title(f"{stage_label}\n{title}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{save_dir}/training_history_per_stage.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Grafik per-stage disimpan: {path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(28, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (jumlah)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    path = f"{save_dir}/confusion_matrix.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix disimpan: {path}")


# ------------------------------------------------------------------
# Evaluasi
# ------------------------------------------------------------------

def evaluate_on_test(model, test_gen, class_names):
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

    report_dict = {}
    all_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    all_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    all_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, cls in enumerate(class_names):
        tp = np.sum((y_true == i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        fp = np.sum((y_true != i) & (y_pred == i))
        tn = np.sum((y_true != i) & (y_pred != i))
        report_dict[cls] = {
            "precision": float(all_precision[i]),
            "recall": float(all_recall[i]),
            "f1_score": float(all_f1[i]),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
            "support": int(np.sum(y_true == i)),
        }

    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    return metrics, report_dict, report_text, y_true, y_pred


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg = Config()
    class_names = sorted(os.listdir(cfg.train_dir))
    num_classes = len(class_names)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"{cfg.output_root}/{cfg.experiment_name}_{timestamp}"
    eval_dir = f"{model_dir}/evaluation"
    os.makedirs(eval_dir, exist_ok=True)

    print(f"Kelas ({num_classes}): {class_names}")

    train_gen, val_gen, test_gen = build_generators(cfg)
    print(f"Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")

    model, base_model = build_model(num_classes, cfg)

    # Stage 1: transfer learning (backbone frozen)
    history_s1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.epochs_stage1,
        callbacks=build_callbacks(model_dir, "best_model_stage1.keras",
                                   "history_stage1.csv", "Stage 1"),
        verbose=0,
    )

    # Stage 2: fine-tuning (unfreeze layer setelah cfg.fine_tune_at)
    base_model.trainable = True
    for layer in base_model.layers[: cfg.fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate / 10),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy", keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"), F1Score(name="f1_score")],
    )

    history_s2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.epochs_stage2,
        callbacks=build_callbacks(model_dir, "best_model_finetuned.keras",
                                   "history_stage2.csv", "Stage 2"),
        verbose=0,
    )

    fit_analysis = analyze_fit(history_s1, history_s2)
    plot_history_per_stage(history_s1, history_s2, eval_dir)

    best_model = keras.models.load_model(
        f"{model_dir}/best_model_finetuned.keras",
        custom_objects={"F1Score": F1Score},
    )
    metrics, report_dict, report_text, y_true, y_pred = evaluate_on_test(
        best_model, test_gen, class_names
    )
    plot_confusion_matrix(y_true, y_pred, class_names, eval_dir)

    print("\nHasil evaluasi test set:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\n" + report_text)

    results = {
        "experiment": cfg.experiment_name,
        "hyperparameters": cfg.__dict__,
        "overfitting_analysis": fit_analysis,
        "overall_metrics": metrics,
        "per_class_metrics": report_dict,
    }
    with open(f"{model_dir}/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{model_dir}/classification_report.txt", "w") as f:
        f.write(report_text)

    print(f"\nSemua output tersimpan di: {model_dir}")


if __name__ == "__main__":
    main()