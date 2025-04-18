# 🌊 Waifu-GAN-Flow

![demo](https://github.com/user-attachments/assets/6b0e7113-a6eb-4bdb-9922-7c64456fad73)

This app can also be deployed on Hugging Face Spaces:  
👉 [waifu-gan-flow on Hugging Face Spaces](https://huggingface.co/spaces/synonym/waifu-gan-flow)

> Smoothly animate full-body anime characters using latent interpolation from [`skytnt/waifu-gan`](https://huggingface.co/skytnt/waifu-gan).

**Waifu-GAN-Flow** is a lightweight ONNX-based toolkit that allows you to:
- 🎨 Generate anime-style full-body characters
- 🔁 Morph between characters using latent space interpolation
- 🎥 Export smooth mp4 animations via Gradio UI

---

## 🚀 Features

- ✅ **ONNX Runtime GPU** for fast inference (CUDA supported)
- 🎲 Latent space **random or seed-based generation**
- 🎥 Interpolate between characters with smooth **transition videos**
- 🖼 Simple **Gradio UI** with multiple tabs: image, batch, video, sequence

---

## 🗂 Directory Structure

```
waifu-gan-flow/
├── models/                  # ONNX models (must be downloaded)
│   ├── g_mapping.onnx
│   └── g_synthesis.onnx
├── outputs/
│   ├── images/              # Output PNGs
│   └── videos/              # Output MP4s
├── fullbody_gan_app.py      # Gradio app (entry point)
├── download_models.py       # Optional: auto-download helper
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 📦 Setup

### 🧪 Option 1: Conda (recommended)

```bash
git clone https://github.com/Mainlst/waifu-gan-flow.git
cd waifu-gan-flow

conda env create -f environment.yml
conda activate waifu-gan
```

### 📦 Option 2: Pip

```bash
pip install -r requirements.txt
```

---

## 📥 Model Download

Manually download the following files from [skytnt/waifu-gan](https://huggingface.co/skytnt/waifu-gan):

- `g_mapping.onnx`
- `g_synthesis.onnx`

Put them into the `models/` directory:

```bash
mkdir models
# Move downloaded ONNX models here
```

Or run the auto-downloader if provided:

```bash
python download_models.py
```

---

## 🌟 Usage

### ✨ Run the app:
```bash
python fullbody_gan_app.py
```

A Gradio UI will launch in your browser with the following tabs:

- **Generate Image**: Random or seed-based generation
- **Batch Generation**: Create multiple images at once
- **Video (Pair)**: Interpolate between two selected waifus
- **Sequence Video**: Auto-generate and morph through a waifu series

---

## 📤 Output Files

- Images → `outputs/images/`
- Videos → `outputs/videos/`

All generated content is saved automatically with seed and psi metadata in the filenames.

---

## 🧠 Model Info

- Source: [`skytnt/waifu-gan`](https://huggingface.co/skytnt/waifu-gan)
- License: **Apache-2.0**
- Input: latent vector `z ∈ ℝ⁵¹²`
- Output: full-body anime-style character image (1024×512)

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## 💖 Credits

- Model: [`skytnt`](https://huggingface.co/skytnt)
- Interface & ONNX Runtime integration: [Mainlst](https://github.com/Mainlst)

---

## ✨ Sample Output

![waifu-sample](https://github.com/user-attachments/assets/a25d1314-3c12-43e5-ad8d-a090f00297b9)

Enjoy generating and animating cute waifus with style 💕

---
