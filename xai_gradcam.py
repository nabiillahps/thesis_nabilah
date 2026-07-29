"""
Identifikasi Pemalsuan Bubuk Cabai Merah Menggunakan Citra Digital
Berbasis EfficientNetV2 dengan Visualisasi Explainable AI

Pipeline lengkap ada 3 tahap :
1. Preprocessing & split data      
2. Training & evaluasi model       
3. Visualisasi Grad-CAM / XAI

Grad-CAM (Explainable AI) untuk model EfficientNetV2S

`find_last_conv_layer` mengasumsikan layer konvolusi terakhir dalam urutan
`backbone.layers` sama dengan layer konvolusi terakhir yang dieksekusi pada
forward pass. Untuk arsitektur dengan skip connection (seperti
EfficientNetV2S), ini umumnya benar tapi bukan jaminan mutlak -- perlu
diverifikasi manual dengan melihat daftar layer (lihat fungsi
`print_last_layers` di bawah) sebelum dipakai untuk laporan akhir.
"""

import os
import zipfile
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers
import matplotlib as mpl
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Konfigurasi
# ------------------------------------------------------------------

@dataclass
class Config:
    model_path: str = os.environ.get("MODEL_PATH", "./model/best_model_finetuned.keras")
    test_dir: str = os.environ.get("TEST_DIR", "./data_split/test")
    output_dir: str = os.environ.get("GRADCAM_OUTPUT_DIR", "./gradcam_output")
    input_shape: tuple = (224, 224)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ------------------------------------------------------------------
# Custom metric (wajib untuk load model yang dilatih dengan F1Score)
# ------------------------------------------------------------------

class F1Score(keras.metrics.Metric):
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


def load_trained_model(model_path):
    return keras.models.load_model(model_path, custom_objects={"F1Score": F1Score})


# ------------------------------------------------------------------
# Cari layer konvolusi terakhir di backbone
# ------------------------------------------------------------------

def find_last_conv_layer(model):
    """
    Cari backbone (sub-model pertama yang punya `.layers`) dan layer
    konvolusi/depthwise-conv terakhir di dalamnya.
    Lihat catatan metodologis di docstring modul soal batasan asumsi ini.
    """
    backbone = None
    for layer in model.layers:
        if hasattr(layer, "layers"):
            backbone = layer
            break
    if backbone is None:
        raise ValueError("Backbone (sub-model) tidak ditemukan di dalam model.")

    last_conv_layer_name = None
    for layer in backbone.layers:
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            last_conv_layer_name = layer.name

    if last_conv_layer_name is None:
        raise ValueError("Tidak ada layer Conv2D/DepthwiseConv2D ditemukan di backbone.")

    return last_conv_layer_name, backbone.name


def print_last_layers(model, backbone_name, n=20):
    """Cetak n layer terakhir backbone -- dipakai untuk verifikasi manual."""
    backbone_model = model.get_layer(backbone_name)
    for layer in backbone_model.layers[-n:]:
        print(f"  {layer.name:<50} | {layer.__class__.__name__}")


# ------------------------------------------------------------------
# Grad-CAM core
# ------------------------------------------------------------------

def get_img_array(img_path, size):
    """Load gambar tanpa normalisasi manual -- model sudah include_preprocessing=True."""
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    return np.expand_dims(array, axis=0)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, backbone_name, pred_index=None):
    """
    Hitung heatmap Grad-CAM.

    Backbone di-treat sebagai sub-model terpisah supaya feature map layer
    konvolusi terakhir bisa diambil sebagai output tambahan. Layer setelah
    backbone (head klasifikasi) dijalankan manual satu per satu di dalam
    GradientTape karena backbone berupa nested model, bukan dijalankan
    langsung lewat model penuh.
    """
    backbone = model.get_layer(backbone_name)
    grad_model = keras.Model(
        inputs=backbone.inputs,
        outputs=[backbone.get_layer(last_conv_layer_name).output, backbone.output],
    )

    post_layers = []
    found = False
    for layer in model.layers:
        if layer.name == backbone_name:
            found = True
            continue
        if found:
            post_layers.append(layer)

    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, backbone_out = grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)

        x = backbone_out[0] if isinstance(backbone_out, list) else backbone_out
        for layer in post_layers:
            x = layer(x, training=False)
        preds = x

        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError("Gradient bernilai None -- cek apakah last_conv_layer_name benar.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), preds.numpy()


def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = keras.utils.load_img(img_path)
    img_array = keras.utils.img_to_array(img)

    heatmap_uint8 = np.uint8(255 * heatmap)
    jet_colors = mpl.colormaps["jet"](np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap_img = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap_img = jet_heatmap_img.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap_arr = keras.utils.img_to_array(jet_heatmap_img)

    superimposed = jet_heatmap_arr * alpha + img_array
    return img_array, jet_heatmap_arr, np.array(keras.utils.array_to_img(superimposed))


# ------------------------------------------------------------------
# Klasifikasi gambar per kelas: benar vs salah prediksi
# ------------------------------------------------------------------

def classify_predictions(model, kelas_dir, kelas, input_shape, class_names):
    """Pisahkan gambar dalam satu folder kelas menjadi prediksi benar/salah."""
    semua_gambar = [
        os.path.join(kelas_dir, f)
        for f in os.listdir(kelas_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    benar, salah = [], []
    for img_path in semua_gambar:
        arr = get_img_array(img_path, size=input_shape)
        preds = model.predict(arr, verbose=0)[0]
        pred_kelas = class_names[np.argmax(preds)]
        if pred_kelas == kelas:
            benar.append((img_path, preds))
        else:
            salah.append((img_path, preds, pred_kelas))
    return semua_gambar, benar, salah


def save_gradcam_figure(img_path, model, last_conv_name, backbone_name, input_shape,
                         save_path, true_label, pred_label, pred_conf, is_correct):
    """Buat dan simpan satu figure Grad-CAM (ground truth | heatmap | superimposed)."""
    arr = get_img_array(img_path, size=input_shape)
    heatmap, _ = make_gradcam_heatmap(arr, model, last_conv_name, backbone_name)
    img_orig, _, superimposed = overlay_gradcam(img_path, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(img_orig.astype("uint8"))
    axes[0].set_title(f"Ground Truth\n{true_label}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap Grad-CAM", fontsize=10)
    axes[1].axis("off")

    status = "Benar" if is_correct else "Salah"
    axes[2].imshow(superimposed.astype("uint8"))
    axes[2].set_title(f"Prediksi: {pred_label} ({pred_conf:.2f}%)\n{status}",
                       fontsize=10, color="black" if is_correct else "red")
    axes[2].axis("off")

    fig.suptitle(f"Grad-CAM - {true_label}", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_gradcam_for_class(model, kelas, cfg: Config, last_conv_name, backbone_name, class_names):
    """
    Generate Grad-CAM untuk semua gambar benar & salah prediksi dalam satu
    kelas, lalu simpan sebagai ZIP terpisah (benar/ dan salah/).
    """
    kelas_dir = os.path.join(cfg.test_dir, kelas)
    semua_gambar, benar, salah = classify_predictions(model, kelas_dir, kelas, cfg.input_shape, class_names)
    akurasi = len(benar) / len(semua_gambar) * 100 if semua_gambar else 0

    print(f"{kelas}: total={len(semua_gambar)} benar={len(benar)} salah={len(salah)} "
          f"akurasi={akurasi:.1f}%")

    for subset, is_correct, subfolder in [(benar, True, "benar"), (salah, False, "salah")]:
        if not subset:
            continue
        temp_dir = f"{cfg.output_dir}/{subfolder}/{kelas}_temp"
        os.makedirs(temp_dir, exist_ok=True)

        for item in subset:
            if is_correct:
                img_path, preds = item
                pred_label = kelas
            else:
                img_path, preds, pred_label = item
            pred_conf = preds[np.argmax(preds)] * 100
            nama_file = os.path.basename(img_path).split(".")[0]
            save_path = f"{temp_dir}/gradcam_{kelas}_{nama_file}.png"
            save_gradcam_figure(img_path, model, last_conv_name, backbone_name, cfg.input_shape,
                                 save_path, kelas, pred_label, pred_conf, is_correct)

        zip_path = f"{cfg.output_dir}/{subfolder}/{kelas}_gradcam_{subfolder}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(temp_dir):
                zf.write(os.path.join(temp_dir, fname), arcname=fname)
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
        print(f"  {len(subset)} gambar ({subfolder}) tersimpan di: {zip_path}")

    return {"total": len(semua_gambar), "benar": len(benar), "salah": len(salah), "akurasi": akurasi}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg = Config()

    class_names = sorted(os.listdir(cfg.test_dir))
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/benar", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/salah", exist_ok=True)

    print(f"Kelas ({len(class_names)}): {class_names}")

    model = load_trained_model(cfg.model_path)
    last_conv_name, backbone_name = find_last_conv_layer(model)
    print(f"Backbone: {backbone_name} | Last conv layer: {last_conv_name}")
    print("20 layer terakhir backbone (verifikasi manual disarankan):")
    print_last_layers(model, backbone_name)

    # Uji cepat 1 gambar sebelum batch penuh
    test_class = class_names[0]
    test_img_dir = os.path.join(cfg.test_dir, test_class)
    test_img_path = os.path.join(test_img_dir, os.listdir(test_img_dir)[0])

    arr = get_img_array(test_img_path, size=cfg.input_shape)
    heatmap, preds = make_gradcam_heatmap(arr, model, last_conv_name, backbone_name)
    pred_class = class_names[np.argmax(preds[0])]
    print(f"Uji cepat -- kelas asli: {test_class}, prediksi: {pred_class} "
          f"({preds[0][np.argmax(preds[0])] * 100:.2f}%)")

    img_orig, _, superimposed = overlay_gradcam(test_img_path, heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_orig.astype("uint8")); axes[0].set_title("Gambar Asli"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Heatmap Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(superimposed.astype("uint8")); axes[2].set_title("Superimposed"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/test_gradcam_single.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Batch penuh per kelas
    summary = {}
    for kelas in class_names:
        summary[kelas] = generate_gradcam_for_class(
            model, kelas, cfg, last_conv_name, backbone_name, class_names
        )

    print(f"\nSelesai. Ringkasan akurasi per kelas tersimpan. Output di: {cfg.output_dir}")
    return summary


if __name__ == "__main__":
    main()