#!/bin/bash
# Quick launch script for Streamlit app

echo "🔬 Launching ABX₃ Bandgap Predictor..."
echo "📍 URL will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "/Users/amirmahboud/Documents/PhD/Writing Paper/ABX3_Bandgap_Predictor"
source /opt/anaconda3/bin/activate ml_perovskite
streamlit run app.py

