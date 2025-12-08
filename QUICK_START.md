# 🚀 Quick Start Guide

## Run the App Locally

Just double-click `launch.sh` or run in Terminal:

```bash
./launch.sh
```

App opens at **http://localhost:8501**

---

## What You'll See

### 1. **Single Prediction**
- Enter A-B-X composition (e.g., Cs-Pb-I)
- Optional: Add synthesis parameters
- Get prediction + **SHAP waterfall plot**
- See top 5 contributing features

### 2. **Batch Prediction**
- Upload CSV with multiple materials
- Download results as CSV
- See summary statistics

### 3. **About Page**
- Model performance metrics
- How it works
- Citation info

---

## Deploy to Streamlit Cloud (For Your Paper)

### Step 1: Create GitHub Repo

```bash
cd "/Users/amirmahboud/Documents/PhD/Writing Paper/ABX3_Bandgap_Predictor"

git init
git add .
git commit -m "ABX3 Bandgap Predictor v1.0"

# Create repo on GitHub (github.com), then:
git remote add origin https://github.com/YOUR_USERNAME/abx3-predictor
git push -u origin main
```

### Step 2: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repo
5. Set main file: `app.py`
6. Click "Deploy"

**✅ Done!** Your app will be live at:
```
https://YOUR-APP-NAME.streamlit.app
```

---

## Screenshots for Paper

**To capture screenshots:**

1. Run `./launch.sh`
2. Navigate to Single Prediction
3. Example: Predict CsPbI₃
4. Save screenshot of SHAP plot
5. Add to paper with caption:

> *"Interactive web application showing SHAP-based feature importance for bandgap prediction. The model identified [feature] as the primary driver, contributing +0.XX eV to the predicted bandgap."*

---

## App Features (Paper-Ready)

✅ **Auto-compute physics features** from composition  
✅ **SHAP explanations** on every prediction  
✅ **Batch processing** for multiple materials  
✅ **Model performance card** (R² = 0.744)  
✅ **Downloadable results**  
✅ **Mobile-responsive UI**  

---

## Troubleshooting

**App won't start?**
```bash
source /opt/anaconda3/bin/activate ml_perovskite
streamlit run app.py
```

**Missing packages?**
```bash
pip install -r requirements.txt
```

**Models not found?**
```bash
./setup.sh
```

---

## Next Steps

1. **Test locally**: Run `./launch.sh`
2. **Try examples**: CsPbI₃, FAPbBr₃, MASnI₃
3. **Deploy to cloud**: Follow GitHub + Streamlit Cloud steps
4. **Add to paper**: Include URL and screenshots

---

**Questions?** Check `README.md` for full documentation.


