
import sys
import pandas as pd
import numpy as np
from inference_utils import load_model_pipeline, predict_single
from feature_engineering import compute_spinel_features
from element_data import ELEMENTS_DATA

def test_spinel_integration():
    print("Testing Spinel Model Integration...")
    
    # 1. Load Model
    try:
        model, manifest, encoders = load_model_pipeline("models/spinel")
        print("✅ Model loaded successfully")
        print(f"   Version: {manifest['pipeline_version']}")
        print(f"   R2: {manifest['performance']['oof_r2']}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Prepare Input (MgAl2O4)
    input_data = {
        'A_element': 'Mg',
        'B_element': 'Al',
        'synthesis_temperature': 900.0,
        'synthesis_time_hours': 12.0,
        'lattice_parameter': 8.05,
        'crystallite_size': 20.0,
        'synthesis_method': 'sol-gel',
        'morphology': 'spherical',
        'bandgap_method': 'Tauc plot'
    }
    
    print("\nInput Data:")
    print(input_data)
    
    # 3. Compute Features
    try:
        features_df = compute_spinel_features(input_data, ELEMENTS_DATA, encoders)
        print("✅ Features computed successfully")
        print("   Features:", features_df.columns.tolist())
        print("   Values:", features_df.values[0])
    except Exception as e:
        print(f"❌ Failed to compute features: {e}")
        return
        
    # 4. Predict
    try:
        result = predict_single(model, features_df, manifest, compute_shap=True)
        print("✅ Prediction successful")
        print(f"   Predicted Bandgap: {result['prediction']:.3f} eV")
        print("   Top SHAP features:")
        for feat, val in result['top_features'][:3]:
            print(f"     - {feat}: {val:.4f}")
    except Exception as e:
        print(f"❌ Failed to predict: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    test_spinel_integration()
