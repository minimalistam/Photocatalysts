#!/bin/bash
# Setup script for ABX3 Bandgap Predictor Streamlit app

echo "🔬 Setting up ABX3 Bandgap Predictor..."

# Create models directory
mkdir -p models

# Copy trained models from training directory
TRAINING_DIR="/Users/amirmahboud/Documents/SINGLE SHOT/rev2 ml/final-submission/Final - clean/data/models"

echo "📦 Copying trained models..."

if [ -f "$TRAINING_DIR/catboost_final.cbm" ]; then
    cp "$TRAINING_DIR/catboost_final.cbm" models/
    echo "  ✅ catboost_final.cbm"
else
    echo "  ❌ catboost_final.cbm not found"
fi

if [ -f "$TRAINING_DIR/feature_manifest.json" ]; then
    cp "$TRAINING_DIR/feature_manifest.json" models/
    echo "  ✅ feature_manifest.json"
else
    echo "  ❌ feature_manifest.json not found"
fi

if [ -f "$TRAINING_DIR/categorical_encoders.pkl" ]; then
    cp "$TRAINING_DIR/categorical_encoders.pkl" models/
    echo "  ✅ categorical_encoders.pkl"
else
    echo "  ❌ categorical_encoders.pkl not found"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Run app: streamlit run app.py"
echo ""


