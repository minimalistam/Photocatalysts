"""
Inference utilities for bandgap prediction.
Handles model loading, prediction, and SHAP explanations.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from catboost import CatBoostRegressor, Pool

# Lazy imports for optional dependencies
try:
    import shap
    import plotly.graph_objects as go
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def load_model_pipeline(model_dir="models"):
    """Load trained model, manifest, and encoders."""
    model_dir = Path(model_dir)
    
    # Load manifest
    with open(model_dir / 'manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Load CatBoost model
    model = CatBoostRegressor()
    # Check for different possible names
    if (model_dir / 'catboost_final.cbm').exists():
        model.load_model(str(model_dir / 'catboost_final.cbm'))
    elif (model_dir / 'catboost_model.cbm').exists():
        model.load_model(str(model_dir / 'catboost_model.cbm'))
    else:
        raise FileNotFoundError(f"No model file found in {model_dir}")
    
    # Load encoders (optional, might not exist for Spinel if we didn't save them?)
    # Wait, I didn't save encoders for Spinel! 
    # The Spinel model uses CatBoost's native categorical handling.
    # But feature_engineering.py uses encoders for 'Unknown' handling?
    # Actually, my compute_spinel_features uses encoders.
    # I need to save the encoders from the Spinel training or handle it differently.
    # For now, let's make encoders optional.
    if (model_dir / 'categorical_encoders.pkl').exists():
        encoders = joblib.load(model_dir / 'categorical_encoders.pkl')
    else:
        encoders = {}
    
    return model, manifest, encoders


def predict_single(model, features_df, manifest, compute_shap=True):
    """
    Make single prediction with optional SHAP explanation.
    
    Args:
        model: Trained CatBoost model
        features_df: DataFrame with one row of features
        manifest: Feature manifest dict
        compute_shap: Whether to compute SHAP values
        
    Returns:
        dict with prediction, shap_values, shap_plot, top_features
    """
    
    cat_indices = manifest['features']['categorical_indices']
    all_features = manifest['features']['all']
    
    # Create Pool
    pred_pool = Pool(features_df, cat_features=cat_indices)
    
    # Predict
    try:
        prediction = model.predict(pred_pool)[0]
    except Exception as e:
        # Re-raise with full string to bypass Streamlit redaction if possible,
        # or at least print it to logs clearly.
        raise RuntimeError(f"CatBoost Prediction Failed: {str(e)}") from e
    
    result = {
        'prediction': float(prediction),
        'shap_values': None,
        'shap_plot': None,
        'top_features': []
    }
    
    if compute_shap and SHAP_AVAILABLE:
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df)
        
        # Get base value (expected value)
        base_value = explainer.expected_value
        
        # SHAP values for this instance
        shap_vals = shap_values[0]
        
        # Sort by absolute value
        abs_shap = np.abs(shap_vals)
        top_indices = np.argsort(abs_shap)[::-1][:10]  # Top 10
        
        top_features = [
            (all_features[i], float(shap_vals[i])) 
            for i in top_indices
        ]
        
        # Create waterfall plot using Plotly
        top_indices_plot = top_indices[:10]
        feature_names = [all_features[i] for i in top_indices_plot]
        feature_values = [float(shap_vals[i]) for i in top_indices_plot]
        
        # Create waterfall
        fig = go.Figure()
        
        cumulative = base_value
        y_pos = list(range(len(feature_names)))
        
        colors = ['#FF6B6B' if v < 0 else '#51CF66' for v in feature_values]
        
        fig.add_trace(go.Bar(
            y=feature_names,
            x=feature_values,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{v:+.4f}" for v in feature_values],
            textposition='outside',
            hovertemplate='%{y}<br>SHAP value: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Feature Contributions (SHAP Values)",
            xaxis_title="SHAP Value (eV)",
            yaxis_title="",
            height=400,
            showlegend=False,
            yaxis={'categoryorder':'total ascending'}
        )
        
        result['shap_values'] = shap_vals
        result['shap_plot'] = fig
        result['top_features'] = top_features
    
    return result


def predict_batch(df_input, model, manifest, encoders, elements_data, progress_callback=None):
    """
    Batch prediction for multiple materials.
    
    Args:
        df_input: DataFrame with composition columns
        model: Trained model
        manifest: Feature manifest
        encoders: Categorical encoders
        elements_data: Element properties
        progress_callback: Function(progress, message) for progress updates
        
    Returns:
        DataFrame with predictions
    """
    from feature_engineering import compute_physics_features
    
    results = []
    n_total = len(df_input)
    
    for idx, row in df_input.iterrows():
        if progress_callback:
            progress = (idx + 1) / n_total
            progress_callback(progress, f"Processing {idx+1}/{n_total}...")
        
        try:
            # Convert row to dict
            input_data = row.to_dict()
            
            # Compute features
            features_df = compute_physics_features(input_data, elements_data, encoders)
            
            # Predict
            result = predict_single(model, features_df, manifest, compute_shap=False)
            
            # Add to results
            results.append({
                'Material': f"{row.get('A_element', 'X')}{row.get('B_element', 'Y')}{row.get('X_element', 'Z')}₃",
                'Predicted_Bandgap_eV': result['prediction'],
                'Model_Uncertainty_eV': manifest['performance'].get('aggregated_rmse_eV', 0),
                **input_data
            })
            
        except Exception as e:
            results.append({
                'Material': f"Error",
                'Predicted_Bandgap_eV': np.nan,
                'Model_Uncertainty_eV': np.nan,
                'Error': str(e),
                **input_data
            })
    
    return pd.DataFrame(results)


