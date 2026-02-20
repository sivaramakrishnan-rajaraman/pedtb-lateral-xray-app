# 🫁 Explainable Pediatric Lateral Chest X-ray Classifier

## 🌐 Live App

Public URL: https://pedtb-lateral-xray-app.streamlit.app/


**Live app:**  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pedtb-lateral-xray-app.streamlit.app/)

This app demonstrates an end-to-end, explainable pipeline on **pediatric lateral chest X-ray classification**:

1. **Custom-trained YOLO11s lateral lung-field detector** detects the **lung field** and crops the lung pixels from the original lateral CXR.
2. **Custom fine-tuned DPN-68 classifier** predicts whether the cropped lateral lungs are **Normal** or show **TB-related signs**.
3. **Explainability (CAM family from Jacobgil PyTorch Grad-CAM)** overlays heatmaps and displays the **heatmap-overlaid** lateral CXR.

**Weights are hosted on Hugging Face Hub and downloaded at runtime:**

---

## 🧭 What is Streamlit? What is Streamlit Cloud?

- **Streamlit** is a Python framework for building interactive web apps in a few lines of code. The code `streamlit_app.py` makes Streamlit render the UI (widgets, images, charts) in the browser.
- **Streamlit Cloud** is a hosted service by Streamlit where we point to our **GitHub repository**, and it *automatically* builds the environment, installs dependencies (**based on the requirements.txt file**), and runs the Streamlit app on their servers.  
  - It runs app’s main file (`streamlit_app.py`).
  - Gives a public URL (here, it is `https://pedtb-lateral-xray-app.streamlit.app/`).

## 🧱 Repository Structure
```bash
pedtb-lateral-xray-app/
├─ streamlit_app.py # Main Streamlit entry point
├─ requirements.txt
├─ runtime.txt # Python version pin for Streamlit Cloud ("3.11")
├─ README.md
├─ .gitignore
└─ src/
├─ hf_utils.py # Helper to download from Hugging Face Hub
├─ tb_model.py 
└─ cam_utils.py # Grad-CAM utilities (uses jacobgil/pytorch-grad-cam)
```
## 🖥️ Local Setup & Run

```bash
# 1) Clone repo
git clone https://github.com/sivaramakrishnan-rajaraman/pedtb-lateral-xray-app.git
cd pedtb-lateral-xray-app

# 2) (Optional) Create a clean Python env (recommend Python 3.11)
#    e.g., using conda or venv

# 3) Install dependencies
pip install -r requirements.txt

# 4) Launch app
streamlit run streamlit_app.py

```