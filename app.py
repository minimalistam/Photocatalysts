"""
ABX₃ Perovskite Bandgap Predictor
==================================
Interactive machine learning tool for predicting bandgaps using trained CatBoost model.

Features:
- Single material prediction with SHAP explanations
- Batch prediction from CSV/Excel
- Auto-computed physics features
- Model interpretability

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
from pathlib import Path
import json
import io
from datetime import datetime


# Import custom modules
from inference_utils import load_model_pipeline, predict_single, predict_batch
from feature_engineering import compute_physics_features, compute_spinel_features, validate_composition
from element_data import ELEMENTS_DATA

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Materials Bandgap Predictor",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        font-weight: 400;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PASSWORD PROTECTION
# ============================================================================

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not continue if password is not correct.

# ============================================================================
# LOAD MODELS (CACHED)
# ============================================================================


def load_models():
    """Load all model pipelines."""
    models = {}
    
    # Perovskite
    try:
        # Prefer subfolder structure
        if (Path("models/perovskite/manifest.json").exists()):
             models['perovskite'] = load_model_pipeline("models/perovskite")
        else:
             models['perovskite'] = load_model_pipeline("models/perovskite") # Will raise if missing
    except Exception as e:
        # Fallback to default 'models' dir for backward compatibility or error
        try:
             # Only fallback if models/manifest.json exists there
             if (Path("models/manifest.json").exists()):
                models['perovskite'] = load_model_pipeline("models")
             else:
                raise e
        except:
            st.error(f"Failed to load Perovskite model: {e}")
            
    # Spinel
    try:
        if (Path("models/spinel/manifest.json").exists()):
            models['spinel'] = load_model_pipeline("models/spinel")
    except Exception as e:
        # st.warning(f"Spinel model not found: {e}")
        pass
        
    return models

# Load models
loaded_models = load_models()

# ============================================================================
# SIDEBAR - INFO & NAVIGATION
# ============================================================================

with st.sidebar:
    st.title("Materials Bandgap Predictor")
    st.caption("Physics-Informed ML for Materials Discovery")
    st.markdown("---")
    
    # Model Selection
    model_type = st.selectbox(
        "Select Material Family",
        ["Perovskite (ABX₃)", "Spinel (AB₂O₄)"]
    )
    
    current_model_key = 'perovskite' if "Perovskite" in model_type else 'spinel'
    
    if current_model_key not in loaded_models:
        st.error(f"Model for {model_type} is not available.")
        st.stop()
        
    model, manifest, encoders = loaded_models[current_model_key]
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["Single Prediction", "Batch Prediction", "About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model Card
    st.subheader("Model Performance")
    perf = manifest['performance']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R²", f"{perf.get('aggregated_r2', 0):.3f}")
        st.metric("RMSE", f"{perf.get('aggregated_rmse_eV', 0):.3f} eV")
    with col2:
        st.metric("MAE", f"{perf.get('aggregated_mae_eV', 0):.3f} eV")
        st.metric("Samples", manifest['n_samples'])
    
    st.caption(f"Version: {manifest['pipeline_version']}")
    st.caption("5-fold cross-validation")
    
    st.markdown("---")
    
    # Citation
    with st.expander("Citation"):
        st.info("To be confirmed")

# ============================================================================
# PAGE: SINGLE PREDICTION
# ============================================================================

if page == "Single Prediction":
    st.markdown(f'<p class="main-header">{model_type} Bandgap Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter composition and synthesis parameters to predict bandgap with SHAP-based feature importance</p>', unsafe_allow_html=True)
    
    # Example materials
    st.markdown("**Quick Examples:**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    # --- State Initialization & Example Helpers ---
    
    # Initialize session state for inputs if not present
    if 'k_A_elem' not in st.session_state:
        st.session_state.k_A_elem = "Cs" if current_model_key == 'perovskite' else "Mg"
    if 'k_B_elem' not in st.session_state:
        st.session_state.k_B_elem = "Pb" if current_model_key == 'perovskite' else "Al"
    if 'k_X_elem' not in st.session_state:
        st.session_state.k_X_elem = "I" if current_model_key == 'perovskite' else "O"
        
    if 'k_synth_method' not in st.session_state: st.session_state.k_synth_method = "Blank"
    if 'k_morphology' not in st.session_state: st.session_state.k_morphology = "Blank"
    if 'k_sample_form' not in st.session_state: st.session_state.k_sample_form = "Blank"
    if 'k_crystal_struct' not in st.session_state: st.session_state.k_crystal_struct = "Blank"
    if 'k_bandgap_type' not in st.session_state: st.session_state.k_bandgap_type = "Blank"
    
    # Helper to bulk update state
    def set_inputs(A, B, X, method="Blank", morph="Blank", form="Blank"):
        st.session_state.k_A_elem = A
        st.session_state.k_B_elem = B
        st.session_state.k_X_elem = X
        st.session_state.k_synth_method = method
        st.session_state.k_morphology = morph
        st.session_state.k_sample_form = form
        # Reset others to Blank to avoid carry-over
        st.session_state.k_crystal_struct = "Blank"
        st.session_state.k_bandgap_type = "Blank"

    if current_model_key == 'perovskite':
        with col_ex1:
            if st.button("Example: CsPbI₃", use_container_width=True): 
                set_inputs("Cs", "Pb", "I")
        with col_ex2:
            if st.button("Example: FAPbBr₃", use_container_width=True): 
                set_inputs("FA", "Pb", "Br")
        with col_ex3:
            if st.button("Example: MASnI₃", use_container_width=True): 
                set_inputs("MA", "Sn", "I")
    else: # Spinel
        with col_ex1:
            if st.button("ZnFe₂O₄ (Combustion)", use_container_width=True, help="High bandgap example (~4.0 eV)"): 
                set_inputs("Zn", "Fe", "O", method="combustion", form="nanoparticles")
        with col_ex2:
            if st.button("ZnFe₂O₄ (Precipitation)", use_container_width=True, help="Low bandgap example (~1.96 eV)"): 
                set_inputs("Zn", "Fe", "O", method="precipitation", form="powder", morph="Nanoscale")
        with col_ex3:
            if st.button("MgFe₂O₄ (Default)", use_container_width=True, help="Typical Ferrite (~2.3 eV)"): 
                set_inputs("Mg", "Fe", "O")
    
    st.markdown("---")
    
    # Two columns: Input | Output
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Composition section
        with st.container():
            st.markdown("#### Composition (Required)")
            
            # Oxidation Defaults
            default_A_ox = 1 if current_model_key=='perovskite' else 2
            default_B_ox = 2 if current_model_key=='perovskite' else 3
            default_X_ox = -1 if current_model_key=='perovskite' else -2
            
            if current_model_key == 'perovskite':
                col_a, col_b, col_x = st.columns(3)
                with col_a:
                    st.caption("A-site element")
                    A_elem = st.text_input("A-site", key="k_A_elem", label_visibility="collapsed")
                    A_ox = st.number_input("A oxidation state", value=default_A_ox, step=1)
                with col_b:
                    st.caption("B-site element")
                    B_elem = st.text_input("B-site", key="k_B_elem", label_visibility="collapsed")
                    B_ox = st.number_input("B oxidation state", value=default_B_ox, step=1)
                with col_x:
                    st.caption("X-site element")
                    X_elem = st.text_input("X-site", key="k_X_elem", label_visibility="collapsed")
                    X_ox = st.number_input("X oxidation state", value=default_X_ox, step=1)
            else: # Spinel
                col_a, col_b, col_x = st.columns(3)
                with col_a:
                    st.caption("A-site element (Tetrahedral)")
                    A_elem = st.text_input("A-site", key="k_A_elem", label_visibility="collapsed")
                    A_ox = st.number_input("A oxidation", value=2, step=1)
                with col_b:
                    st.caption("B-site element (Octahedral)")
                    B_elem = st.text_input("B-site", key="k_B_elem", label_visibility="collapsed")
                    B_ox = st.number_input("B oxidation", value=3, step=1)
                with col_x:
                    st.caption("X-site element (Anion)")
                    # X is fixed to O for Spinel, but kept in state for consistency
                    # If we disable it, we should ensure state has "O".
                    X_elem = st.text_input("X-site", value="O", disabled=True, label_visibility="collapsed") 
                    X_ox = st.number_input("X oxidation", value=-2, disabled=True, step=1)
        
        st.markdown("---")
        
        # Synthesis parameters
        with st.expander("Synthesis Conditions", expanded=True):
            # Allow blank inputs for Temperature and Time
            c1, c2 = st.columns(2)
            with c1:
                # Use value=None to allow empty
                synth_temp = st.number_input("Synthesis Temperature (°C)", value=None, step=10.0, placeholder="blank")
            
            synth_time = None
            if current_model_key == 'perovskite':
                with c2:
                    synth_time = st.number_input("Synthesis Time (hours)", value=None, step=0.5, placeholder="blank")

            if current_model_key == 'perovskite':
                 # Strict User Lists
                 
                 # Synthesis Method
                 # solution-based (hot-injection), solution-based, solid-state, post-synthesis modification, hydrothermal, solvothermal, microwave-assisted, solution-based (etching), templated synthesis, vapor-assisted, thin-film deposition, vapor-phase, sol-gel, mechanochemical, coprecipitation, sonochemical, Blank
                 s_methods = [
                     "solution-based (hot-injection)", "solution-based", "solid-state", "post-synthesis modification",
                     "hydrothermal", "solvothermal", "microwave-assisted", "solution-based (etching)",
                     "templated synthesis", "vapor-assisted", "thin-film deposition", "vapor-phase",
                     "sol-gel", "mechanochemical", "coprecipitation", "sonochemical", "Blank"
                 ]
                 # For Perovskite, we haven't implemented comprehensive examples for synthesis yet, 
                 # so we can leave as default index or add keys if we want consistency.
                 # Let's add key for consistency.
                 if 'k_synth_method' not in st.session_state or st.session_state.k_synth_method not in s_methods:
                     st.session_state.k_synth_method = "Blank"
                     
                 synth_method = st.selectbox("Synthesis Method", s_methods, key="k_synth_method")
                 
                 # Crystal Structure
                 c_structs = [
                     "cubic", "orthorhombic", "tetragonal", "hexagonal", "trigonal", 
                     "rhombohedral", "monoclinic", "mixed-phase", "triclinic", "Blank"
                 ]
                 if 'k_crystal_struct' not in st.session_state or st.session_state.k_crystal_struct not in c_structs:
                     st.session_state.k_crystal_struct = "Blank"
                 crystal_struct = st.selectbox("Crystal Structure", c_structs, key="k_crystal_struct")

                 # Sample Form
                 s_forms = ["nano", "powder", "crystal", "film", "bulk", "mixed", "composite", "Blank"]
                 if 'k_sample_form' not in st.session_state or st.session_state.k_sample_form not in s_forms:
                     st.session_state.k_sample_form = "Blank"
                 sample_form = st.selectbox("Sample Form", s_forms, key="k_sample_form")
                 
                 # Bandgap Type
                 bg_types = ["Direct", "Indirect", "Blank"]
                 if 'k_bandgap_type' not in st.session_state or st.session_state.k_bandgap_type not in bg_types:
                     st.session_state.k_bandgap_type = "Blank"
                 bandgap_type = st.selectbox("Bandgap Type", bg_types, key="k_bandgap_type")
                 
                 # Not used:
                 morphology = "Unknown"
                 phase_purity = "Unknown"

            else:
                # Spinel Inputs (User Specified)
                
                # Synthesis Method
                spinel_methods = [
                    "combustion", "sol-gel", "precipitation", "solid-state", 
                    "Other", "Vapor/Physical", "hydrothermal", "Blank"
                ]
                synth_method = st.selectbox("Synthesis Method", spinel_methods, key="k_synth_method")

                # Morphology 
                spinel_morphs = [
                    "Nanoscale", "agglomerated", "Bulk/Granular", "spherical", 
                    "Geometric/Shaped", "Porous", "mixed morphology", "Blank"
                ]
                morphology = st.selectbox("Morphology", spinel_morphs, key="k_morphology")
                
                # Crystal Structure - Fixed to Cubic
                st.text_input("Crystal Structure", value="Cubic", disabled=True)
                crystal_struct = "Cubic"
                
                # Sample Form
                spinel_forms = [
                    "powder", "nanoparticles", "bulk", "thin film", 
                    "nanowire array", "single crystal", "nanocrystals", "Blank"
                ]
                sample_form = st.selectbox("Sample Form", spinel_forms, key="k_sample_form")
                
                # Bandgap Type
                spinel_bg_types = ["Direct", "Indirect", "Blank"]
                bandgap_type = st.selectbox("Bandgap Type", spinel_bg_types, index=2)
                
                # Phase Purity
                spinel_purity = ["Pure", "Impure", "Blank"]
                phase_purity = st.selectbox("Phase Purity", spinel_purity, index=2)

        st.markdown("---")
        
        # Predict button
        predict_btn = st.button("Predict Bandgap", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Prediction Results")
        
        if predict_btn:
            with st.spinner("Computing physics features and predicting..."):
                # Prepare input
                
                # Helper to handle "Blank"
                def clean_cat(val):
                    if val == "Blank" or val is None:
                        return "Unknown"
                    return val.lower() if isinstance(val, str) and val in ["Direct", "Indirect"] else val # Handle Bandgap casing if needed

                # Handle numeric blanks
                s_temp = synth_temp if synth_temp is not None else np.nan
                s_time = synth_time if synth_time is not None else np.nan
                
                # Correct Crystal Structure / Sample Form / Method passing
                # The encoder expects specific strings.
                # If "Blank" -> "Unknown"
                
                input_data = {
                    'A_element': A_elem.strip(),
                    'A_oxidation': A_ox,
                    'B_element': B_elem.strip(),
                    'B_oxidation': B_ox,
                    'X_element': X_elem.strip(),
                    'X_oxidation': X_ox,
                    'synthesis_temperature': s_temp,
                    'synthesis_time_hours': s_time,
                    'crystal_structure': clean_cat(crystal_struct),
                    'sample_form': clean_cat(sample_form),
                    'synthesis_method': clean_cat(synth_method),
                    'morphology': clean_cat(morphology),
                    'bandgap_type': clean_cat(bandgap_type),
                    'phase_purity': clean_cat(phase_purity)
                }
                
                # Validate Composition
                is_valid, warnings = validate_composition(input_data, ELEMENTS_DATA, model_type=current_model_key)
                
                if not is_valid:
                    for w in warnings:
                        st.error(w)
                    st.stop()
                
                # Show non-critical warnings
                for w in warnings:
                    st.warning(w)
                
                if is_valid:
                    # Compute features
                    if current_model_key == 'perovskite':
                        features_df = compute_physics_features(input_data, ELEMENTS_DATA, encoders, manifest)
                    else:
                        features_df = compute_spinel_features(input_data, ELEMENTS_DATA, encoders, manifest)
                    
                    # Predict with SHAP
                    try:
                        result = predict_single(model, features_df, manifest, compute_shap=True)
                    except Exception as e:
                        st.error(f"Prediction Error: {str(e)}")
                        st.stop()
                    
                    # Display prediction
                    st.success("Prediction Complete")
                    
                    # Big number
                    st.metric(
                        label="Predicted Bandgap",
                        value=f"{result['prediction']:.3f} eV",
                        delta=f"±{perf.get('aggregated_rmse_eV', 0):.3f} eV",
                        delta_color="off"
                    )
                    
                    st.caption("Uncertainty represents model RMSE from cross-validation")
                    
                    st.markdown("---")
                    
                    # SHAP explanation
                    st.markdown("#### Feature Contributions (SHAP Analysis)")
                    if result['shap_plot']:
                        st.plotly_chart(result['shap_plot'], use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Download
                    result_df = pd.DataFrame([{
                        'Material': f"{A_elem}{B_elem}{X_elem}₃" if current_model_key == 'perovskite' else f"{A_elem}{B_elem}₂O₄",
                        'Predicted_Bandgap_eV': result['prediction'],
                        'Model_Uncertainty_eV': perf.get('aggregated_rmse_eV', 0),
                        **input_data
                    }])
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Result (CSV)",
                        data=csv,
                        file_name="bandgap_prediction.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ============================================================================
# PAGE: BATCH PREDICTION
# ============================================================================

elif page == "Batch Prediction":
    st.markdown(f'<p class="main-header">{model_type} Batch Prediction</p>', unsafe_allow_html=True)
    
    # Template Download
    template_path = Path("data/batch_template.xlsx")
    if template_path.exists():
        with open(template_path, "rb") as f:
            st.download_button(
                label="Download Template (Excel)",
                data=f,
                file_name="batch_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Excel template with dropdown menus for categorical features."
            )
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)
            
        st.success(f"Loaded {len(df_input)} materials")
        
        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Processing..."):
                # Custom batch loop to handle feature engineering selection
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in df_input.iterrows():
                    progress_bar.progress((idx + 1) / len(df_input))
                    try:
                        input_data = row.to_dict()
                        # Add defaults if missing
                        if 'A_oxidation' not in input_data: input_data['A_oxidation'] = 2
                        if 'B_oxidation' not in input_data: input_data['B_oxidation'] = 3
                        if 'X_element' not in input_data: input_data['X_element'] = 'O'
                        
                        if current_model_key == 'perovskite':
                            features_df = compute_physics_features(input_data, ELEMENTS_DATA, encoders, manifest)
                        else:
                            features_df = compute_spinel_features(input_data, ELEMENTS_DATA, encoders, manifest)
                            
                        result = predict_single(model, features_df, manifest, compute_shap=False)
                        
                        results.append({
                            'Material': f"Row {idx+1}",
                            'Predicted_Bandgap_eV': result['prediction'],
                            **input_data
                        })
                    except Exception as e:
                        results.append({'Error': str(e)})
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                csv = results_df.to_csv(index=False)
                st.download_button("Download Results", csv, "batch_results.csv", "text/csv")

# ============================================================================
# PAGE: ABOUT
# ============================================================================

elif page == "About":
    st.markdown('<p class="main-header">About This Tool</p>', unsafe_allow_html=True)
    
    if current_model_key == 'perovskite':
        st.markdown(f"""
        ### Model: CatBoost Regressor
        
        **Description:**
        The perovskite model was trained mainly on conventional ABX₃ perovskite structures including both inorganic oxide perovskites (e.g., titanates, ferrites, manganites, and niobates) and hybrid organic-inorganic halide perovskites. Therefore, within these classes, the model demonstrates robust predictive accuracy.

        **Training Data:**
        - **Samples**: {manifest['n_samples']}
        
        **Performance:**
        | Model | R² | RMSE (eV) | MAE (eV) |
        | :--- | :--- | :--- | :--- |
        | CatBoost | {perf.get('aggregated_r2', 0):.2f} | {perf.get('aggregated_rmse_eV', 0):.2f} | {perf.get('aggregated_mae_eV', 0):.2f} |
        """)
        
    else: # Spinel
        st.markdown(f"""
        ### Model: CatBoost Regressor
        
        **Description:**
        This model was mainly trained on Ferrites ($MFe_2O_4$) and Aluminates ($MAl_2O_4$). As a result, the model shows robust predictive performance within these classes of materials.
        
        **Training Data:**
        - **Samples**: {manifest['n_samples']}
        
        **Performance:**
        | Model | R² | RMSE (eV) | MAE (eV) |
        | :--- | :--- | :--- | :--- |
        | CatBoost | {perf.get('aggregated_r2', 0):.2f} | {perf.get('aggregated_rmse_eV', 0):.2f} | {perf.get('aggregated_mae_eV', 0):.2f} |
        """)
    
    st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("© 2025 Amir Mahboud | RMIT University")
