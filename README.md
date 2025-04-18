# 🌊 Waifu-GAN-Flow
![2025-04-17_23-47-10_1](https://github.com/user-attachments/assets/6b0e7113-a6eb-4bdb-9922-7c64456fad73)

This app can be published and run on Hugging Face Spaces:

👉 [waifu-gan-flow on Hugging Face Spaces](https://huggingface.co/spaces/synonym/waifu-gan-flow)

> Smoothly animate full-body anime characters using latent interpolation from skytnt/waifu-gan.

**Waifu-GAN-Flow** is a lightweight toolkit built on top of [`skytnt/waifu-gan`](https://huggingface.co/skytnt/waifu-gan) that lets you:
- 🎨 Generate high-quality anime-style characters
- 🔁 Morph seamlessly between multiple waifus
- 🎥 Export smooth latent-space videos

Whether you're creating transitions from random seeds or interpolating encoded images, **flow** your waifus in style!

<p align="center">
  <img src="https://huggingface.co/skytnt/waifu-gan/resolve/main/sample.png" width="512"/>
</p>

---

## 🚀 Features

- ✅ **ONNX Runtime** for fast & portable inference
- 🎲 Latent space **random or seed-based generation**
- 🎥 Smooth **video interpolation** between two latent vectors
- 🖼 Simple **Gradio UI** for testing

---

## 🗂 Directory Structure

```
waifu-gan-onnx/
├── models/              # ONNX models (not included here — see below)
│   ├── g_mapping.onnx
│   └── g_synthesis.onnx
├── scripts/
├── outputs/
│   ├── images/                 # Saved PNGs
│   └── videos/                 # Saved MP4s
├── fullbody_gan_app.py     # Gradio interface
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
```

---

## 📦 Setup

### 🔧 1. Clone & prepare environment
```bash
git clone https://github.com/Mainlst/waifu-gan-onnx.git
cd waifu-gan-onnx

# (recommended) create a new environment
conda env create -f environment.yml
conda activate waifu-gan
```

Or for pip users:
```bash
pip install -r requirements.txt
```

### 📥 2. Download ONNX models

Download the following files from [Hugging Face - skytnt/waifu-gan](https://huggingface.co/skytnt/waifu-gan):

- `g_mapping.onnx`
- `g_synthesis.onnx`

Place them into the `./models/` folder.

---

## 🌟 How to Use

### ✨ Run the app:
```bash
python fullbody_gan_app.py
```

A **Gradio UI** will appear in your browser.

### ✅ Features:

- **Generate**: Random or seed-based waifu generation
- **Interpolate**: Select two waifus and create a smooth transition video
- **Exported files**:
  - `outputs/images/*.png`
  - `outputs/videos/*.mp4`

---

## 🧠 Model Info

- Source: [https://huggingface.co/skytnt/waifu-gan](https://huggingface.co/skytnt/waifu-gan)
- License: **Apache-2.0**
- Input: latent vector `z` ∈ ℝ⁵¹²
- Output: full-body anime character image (1024×512)

---

## 📄 License

Apache License 2.0  
See [LICENSE](LICENSE) for full text.

---

## 💖 Credit

- Model by [`skytnt`](https://huggingface.co/skytnt)
- ONNX runtime interface and Gradio UI by [YourName](https://github.com/yourname)

Enjoy generating cute waifus! ✨
