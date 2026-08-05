# ♻️ Garbage Classification Dashboard

A deep learning web application built with **Streamlit** and **TensorFlow** that classifies waste images into 12 categories and provides disposal recommendations.

---

## 🚀 Live Demo

> Deploy on [Streamlit Cloud](https://streamlit.io/cloud) — see deployment steps below.

---

## 📂 Project Structure

```
your-repo/
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Python dependencies
├── garbage_cnn_model.keras # Trained CNN model
├── .streamlit/
│   └── config.toml         # Theme configuration
├── .gitignore
└── README.md
```

---

## 🗂️ Waste Categories (12 Classes)

| Emoji | Class | Disposal Bin |
|-------|-------|-------------|
| 🔋 | Battery | Hazardous Waste |
| 🍃 | Biological | Organic / Compost |
| 🟤 | Brown Glass | Glass Recycling |
| 📦 | Cardboard | Paper Recycling |
| 👕 | Clothes | Textile Donation |
| 🟢 | Green Glass | Glass Recycling |
| ⚙️ | Metal | Metal Recycling |
| 📄 | Paper | Paper Recycling |
| 🧴 | Plastic | Plastic Recycling |
| 👟 | Shoes | Textile Donation |
| 🗑️ | Trash | General Waste |
| ⬜ | White Glass | Glass Recycling |

---

## 🧠 Model Architecture

- **Type**: Convolutional Neural Network (CNN)
- **Input Size**: 256 × 256 × 3 (RGB)
- **Conv Blocks**: 3 blocks (Conv2D + BatchNorm + MaxPooling)
- **Regularization**: L2 + Dropout
- **Output**: 12 classes (Softmax)
- **Model File**: `garbage_cnn_model.keras`

---

## 🛠️ Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a Virtual Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## ☁️ Deployment on Streamlit Cloud

### Step 1 — Push to GitHub
Make sure all files are committed and pushed:
```bash
git add .
git commit -m "Initial commit - Garbage Classification Dashboard"
git push origin main
```

> ⚠️ **If `garbage_cnn_model.keras` exceeds 100MB**, use Git LFS:
> ```bash
> git lfs install
> git lfs track "*.keras"
> git add .gitattributes
> git add garbage_cnn_model.keras
> git commit -m "Add model with Git LFS"
> git push origin main
> ```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **"New app"**
3. Connect your GitHub account
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`
5. Click **"Deploy!"**

---

## 📦 Dependencies

| Package | Version |
|---------|---------|
| streamlit | ≥ 1.32.0 |
| tensorflow | ≥ 2.15.0 |
| numpy | ≥ 1.24.0 |
| Pillow | ≥ 10.0.0 |
| plotly | ≥ 5.18.0 |
| pandas | ≥ 2.0.0 |

---

## 📸 Features

- 📤 Upload waste images (JPG, PNG, WEBP)
- 🔍 Real-time classification with confidence score
- 📊 Probability distribution chart (all 12 classes)
- 🎯 Confidence gauge meter
- 💡 Disposal tips per predicted class
- 🌙 Dark green themed UI

---

## 📄 License

This project is for educational purposes. Feel free to use and modify it.

---

<p align="center">Built with ❤️ using TensorFlow & Streamlit</p>
