#🔬 ABX₃ Perovskite Bandgap Predictor

Interactive Streamlit web app for predicting perovskite bandgaps with SHAP explanations.

## 🚀 Quick Start

### 1. Setup

```bash
cd "/Users/amirmahboud/Documents/PhD/Writing Paper/ABX3_Bandgap_Predictor"

# Install dependencies
pip install -r requirements.txt
```

### 2. Copy Trained Models

Copy your trained model files to `models/` directory:

```bash
mkdir models
cp "/Users/amirmahboud/Documents/SINGLE SHOT/rev2 ml/final-submission/Final - clean/data/models/"* models/
```

**Required files:**
- `catboost_final.cbm`
- `feature_manifest.json`
- `categorical_encoders.pkl`

### 3. Run App

```bash
streamlit run app.py
```

App will open at `http://localhost:8501`

---

## 📂 Project Structure

```
ABX3_Bandgap_Predictor/
├── app.py                      # Main Streamlit app
├── inference_utils.py          # Model loading & prediction
├── feature_engineering.py      # Auto-compute physics features
├── element_data.py             # Periodic table data
├── models/                     # Trained model files (you copy these)
│   ├── catboost_final.cbm
│   ├── feature_manifest.json
│   └── categorical_encoders.pkl
├── data/
│   └── batch_template.csv      # Template for batch upload
├── requirements.txt
└── README.md
```

---

## 🌐 Deploy to Streamlit Cloud (Public)

### 1. Create GitHub Repository

```bash
git init
git add .
git commit -m "Initial commit: ABX3 Bandgap Predictor"
git remote add origin https://github.com/YOUR_USERNAME/abx3-bandgap-predictor
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file: `app.py`
6. Click "Deploy"

**Done!** Your app will be live at `https://YOUR_APP_NAME.streamlit.app`

---

## 🎯 Features

✅ **Single Prediction**
- Auto-compute physics features from A-B-X composition
- SHAP waterfall plot for interpretability
- Model uncertainty estimates

✅ **Batch Prediction**
- Upload CSV/Excel with multiple materials
- Progress tracking
- Downloadable results

✅ **Model Card**
- R² = 0.744
- RMSE = 0.33 eV
- Trained on 570 materials

---

## 📝 Usage in Paper

### Screenshots for Paper

```bash
# Start app
streamlit run app.py

# Take screenshots of:
# 1. Single prediction with SHAP plot
# 2. Batch prediction interface
# 3. Model performance card
```

### Citing the Tool

```
This work utilized an interactive machine learning tool for 
bandgap prediction, available at: https://YOUR_APP_URL.streamlit.app
```

---

## ⚠️ Notes

- App requires ~2GB memory (for SHAP calculations)
- SHAP computation takes ~5-10 seconds per prediction
- Batch predictions without SHAP are faster
- All data stays local (nothing sent to external servers)

---

## 📧 Contact

Amir Mahboud  
[amir.mahboud@university.edu](mailto:amir.mahboud@university.edu)

---

## 🔄 Updates

**v1.0** - Initial release with:
- CatBoost model
- SHAP explanations
- Batch processing


