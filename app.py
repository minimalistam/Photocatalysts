"""
ABX₃ Perovskite Bandgap Predictor
==================================
Interactive machine learning tool for predicting bandgaps using trained CatBoost model.

Features:
- Single material prediction with SHAP explanations
- Batch prediction from CSV/Excel
- Auto-computed physics features
- Model interpretability
- User feedback collection

Author: Amir Mahboud
Version: v1.0
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

# Custom CSS for scientific look
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
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not continue if password is not correct.

# ============================================================================
# LOAD MODELS (CACHED)
# ============================================================================

@st.cache_resource
def load_models():
    """Load all model pipelines."""
    models = {}
    
    # Perovskite
    try:
        models['perovskite'] = load_model_pipeline("models/perovskite")
    except Exception as e:
        # Fallback to default 'models' dir for backward compatibility
        try:
            models['perovskite'] = load_model_pipeline("models")
        except:
            st.error(f"Failed to load Perovskite model: {e}")
            
    # Spinel
    try:
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
        ["Single Prediction", "Batch Prediction", "Feedback & Validation", "About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model Card
    st.subheader("Model Performance")
    perf = manifest['performance']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R²", f"{perf['oof_r2']:.3f}")
        st.metric("RMSE", f"{perf['oof_rmse_eV']:.3f} eV")
    with col2:
        st.metric("MAE", f"{perf['oof_mae_eV']:.3f} eV")
        st.metric("Samples", manifest['n_samples'])
    
    st.caption(f"Version: {manifest['pipeline_version']}")
    st.caption("5-fold cross-validation")
    
    st.markdown("---")
    
    # Citation
    with st.expander("Citation"):
        st.code("""
@article{mahboud2025materials,
  title={Physics-Informed Machine Learning 
         for Bandgap Prediction},
  author={Mahboud, Amir et al.},
  journal={In preparation},
  year={2025}
}
        """, language="bibtex")

# ============================================================================
# PAGE: SINGLE PREDICTION
# ============================================================================

if page == "Single Prediction":
    st.markdown(f'<p class="main-header">{model_type} Bandgap Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter composition and synthesis parameters to predict bandgap with SHAP-based feature importance</p>', unsafe_allow_html=True)
    
    # Example materials
    st.markdown("**Quick Examples:**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    if current_model_key == 'perovskite':
        with col_ex1:
            if st.button("Example: CsPbI₃", use_container_width=True): st.session_state.example = "CsPbI3"
        with col_ex2:
            if st.button("Example: FAPbBr₃", use_container_width=True): st.session_state.example = "FAPbBr3"
        with col_ex3:
            if st.button("Example: MASnI₃", use_container_width=True): st.session_state.example = "MASnI3"
    else: # Spinel
        with col_ex1:
            if st.button("Example: MgAl₂O₄", use_container_width=True): st.session_state.example = "MgAl2O4"
        with col_ex2:
            if st.button("Example: ZnFe₂O₄", use_container_width=True): st.session_state.example = "ZnFe2O4"
        with col_ex3:
            if st.button("Example: CoAl₂O₄", use_container_width=True): st.session_state.example = "CoAl2O4"
    
    st.markdown("---")
    
    # Two columns: Input | Output
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Composition section
        with st.container():
            st.markdown("#### Composition (Required)")
            
            # Defaults
            if current_model_key == 'perovskite':
                default_A, default_B, default_X = "Cs", "Pb", "I"
                default_A_ox, default_B_ox, default_X_ox = 1, 2, -1
            else:
                default_A, default_B, default_X = "Mg", "Al", "O" # X is unused/fixed
                default_A_ox, default_B_ox, default_X_ox = 2, 3, -2
            
            # Check for example
            if 'example' in st.session_state:
                ex = st.session_state.example
                if ex == "CsPbI3": default_A, default_B, default_X = "Cs", "Pb", "I"
                elif ex == "FAPbBr3": default_A, default_B, default_X = "FA", "Pb", "Br"
                elif ex == "MASnI3": default_A, default_B, default_X = "MA", "Sn", "I"
                elif ex == "MgAl2O4": default_A, default_B = "Mg", "Al"
                elif ex == "ZnFe2O4": default_A, default_B = "Zn", "Fe"
                elif ex == "CoAl2O4": default_A, default_B = "Co", "Al"
                del st.session_state.example
            
            if current_model_key == 'perovskite':
                col_a, col_b, col_x = st.columns(3)
                with col_a:
                    st.caption("A-site element")
                    A_elem = st.text_input("A-site", value=default_A, label_visibility="collapsed")
                    A_ox = st.number_input("A oxidation state", value=default_A_ox, step=1)
                with col_b:
                    st.caption("B-site element")
                    B_elem = st.text_input("B-site", value=default_B, label_visibility="collapsed")
                    B_ox = st.number_input("B oxidation state", value=default_B_ox, step=1)
                with col_x:
                    st.caption("X-site element")
                    X_elem = st.text_input("X-site", value=default_X, label_visibility="collapsed")
                    X_ox = st.number_input("X oxidation state", value=default_X_ox, step=1)
            else: # Spinel
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption("A-site element (Tetrahedral)")
                    A_elem = st.text_input("A-site", value=default_A, label_visibility="collapsed")
                    # A_ox usually +2
                with col_b:
                    st.caption("B-site element (Octahedral)")
                    B_elem = st.text_input("B-site", value=default_B, label_visibility="collapsed")
                    # B_ox usually +3
                
                # Hidden inputs for compatibility
                A_ox, B_ox, X_elem, X_ox = 2, 3, "O", -2
        
        st.markdown("---")
        
        # Synthesis parameters
        with st.expander("Synthesis & Structure (Required/Optional)", expanded=True):
            if current_model_key == 'spinel':
                st.info("For Spinels, Lattice Parameter is CRITICAL.")
                lattice_param = st.number_input("Lattice Parameter (Å)", value=8.08, step=0.01, format="%.4f")
                cryst_size = st.number_input("Crystallite Size (nm)", value=20.0, step=1.0)
            
            synth_temp = st.number_input("Synthesis Temperature (°C)", value=600.0, step=10.0)
            synth_time = st.number_input("Synthesis Time (hours)", value=12.0, step=0.5)
            
            if current_model_key == 'perovskite':
                anneal_temp = st.number_input("Annealing Temperature (°C)", value=100.0, step=10.0)
                anneal_time = st.number_input("Annealing Time (hours)", value=1.0, step=0.5)
            else:
                anneal_temp, anneal_time = 0.0, 0.0
        
        # Categorical
        with st.expander("Method & Morphology", expanded=False):
            synth_method = st.selectbox("Synthesis Method", 
                ["Unknown", "Solution", "Solid-State", "Combustion", "Hydrothermal", "Sol-Gel", "Co-precipitation"])
            morphology = st.selectbox("Morphology", 
                ["Unknown", "Nanoparticles", "Bulk", "Film", "Spherical", "Agglomerated"])
            
            if current_model_key == 'perovskite':
                crystal_struct = st.selectbox("Crystal Structure", ["Cubic", "Tetragonal", "Orthorhombic"])
                sample_form = st.selectbox("Sample Form", ["Powder", "Film", "Single Crystal"])
                bandgap_type = st.selectbox("Bandgap Type", ["Direct", "Indirect"])
            else:
                crystal_struct, sample_form, bandgap_type = "Cubic", "Powder", "Direct"
                bandgap_method = st.selectbox("Bandgap Method", ["Tauc plot", "UV-Vis", "DRS"])

        st.markdown("---")
        
        # Predict button
        predict_btn = st.button("Predict Bandgap", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Prediction Results")
        
        if predict_btn:
            with st.spinner("Computing physics features and predicting..."):
                # Prepare input
                input_data = {
                    'A_element': A_elem.strip(),
                    'A_oxidation': A_ox,
                    'B_element': B_elem.strip(),
                    'B_oxidation': B_ox,
                    'X_element': X_elem.strip(),
                    'X_oxidation': X_ox,
                    'synthesis_temperature': synth_temp,
                    'synthesis_time_hours': synth_time,
                    'annealing_temperature': anneal_temp,
                    'annealing_time_hours': anneal_time,
                    'crystal_structure': crystal_struct,
                    'sample_form': sample_form,
                    'synthesis_method': synth_method,
                    'morphology': morphology,
                    'bandgap_type': bandgap_type
                }
                
                if current_model_key == 'spinel':
                    input_data['lattice_parameter'] = lattice_param
                    input_data['crystallite_size'] = cryst_size
                    input_data['bandgap_method'] = bandgap_method
                
                # Validate (Basic)
                is_valid = True
                if not A_elem or not B_elem:
                    st.error("Please enter elements.")
                    is_valid = False
                
                if is_valid:
                    # Compute features
                    if current_model_key == 'perovskite':
                        features_df = compute_physics_features(input_data, ELEMENTS_DATA, encoders)
                    else:
                        features_df = compute_spinel_features(input_data, ELEMENTS_DATA, encoders)
                    
                    # Predict with SHAP
                    result = predict_single(model, features_df, manifest, compute_shap=True)
                    
                    # Display prediction
                    st.success("Prediction Complete")
                    
                    # Big number
                    st.metric(
                        label="Predicted Bandgap",
                        value=f"{result['prediction']:.3f} eV",
                        delta=f"±{perf['oof_rmse_eV']:.3f} eV",
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
                        'Model_Uncertainty_eV': perf['oof_rmse_eV'],
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
                            features_df = compute_physics_features(input_data, ELEMENTS_DATA, encoders)
                        else:
                            features_df = compute_spinel_features(input_data, ELEMENTS_DATA, encoders)
                            
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
# PAGE: FEEDBACK & VALIDATION
# ============================================================================

elif page == "Feedback & Validation":
    st.markdown('<p class="main-header">User Feedback & Experimental Validation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Submit experimental results to improve model accuracy</p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    **Help us improve the {model_type} model!** 
    
    If you have experimentally measured bandgaps for materials you've predicted, 
    please submit them here. Your data will be used to retrain and improve the model.
    """)
    
    st.markdown("---")
    
    with st.form("feedback_form"):
        st.subheader("Material Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fb_material = st.text_input("Material Formula", 
                                       help="Full chemical formula (e.g., CsPbI3 or MgAl2O4)")
            fb_A = st.text_input("A-site element")
            fb_B = st.text_input("B-site element")
            if current_model_key == 'perovskite':
                fb_X = st.text_input("X-site element")
            else:
                fb_X = "O" # Spinel always Oxide for now
            
        with col2:
            fb_predicted = st.number_input("Predicted Bandgap (eV)", 
                                          min_value=0.0, max_value=10.0, 
                                          value=0.0, step=0.01)
            fb_experimental = st.number_input("Experimental Bandgap (eV)", 
                                             min_value=0.0, max_value=10.0, 
                                             value=0.0, step=0.01)
        
        st.subheader("Additional Information (Optional)")
        fb_synthesis = st.text_area("Synthesis Details")
        fb_notes = st.text_area("Notes or Comments")
        fb_email = st.text_input("Email (optional, for follow-up)")
        
        submitted = st.form_submit_button("Submit Feedback", type="primary")
        
        if submitted:
            if fb_material and fb_experimental > 0:
                # Save to CSV
                feedback_data = {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': model_type,
                    'material': fb_material,
                    'A_element': fb_A,
                    'B_element': fb_B,
                    'X_element': fb_X,
                    'predicted_bandgap_eV': fb_predicted,
                    'experimental_bandgap_eV': fb_experimental,
                    'synthesis_details': fb_synthesis,
                    'notes': fb_notes,
                    'email': fb_email
                }
                
                # Append to feedback file
                feedback_file = Path('data/user_feedback.csv')
                feedback_file.parent.mkdir(exist_ok=True)
                
                df_feedback = pd.DataFrame([feedback_data])
                
                if feedback_file.exists():
                    df_existing = pd.read_csv(feedback_file)
                    df_feedback = pd.concat([df_existing, df_feedback], ignore_index=True)
                
                df_feedback.to_csv(feedback_file, index=False)
                
                st.success("Thank you! Your feedback has been recorded.")
                st.info(f"Total submissions: {len(df_feedback)}")
            else:
                st.error("Please provide at least material formula and experimental bandgap.")
    
    st.markdown("---")
    
    # Show statistics if feedback exists
    feedback_file = Path('data/user_feedback.csv')
    if feedback_file.exists():
        df_feedback = pd.read_csv(feedback_file)
        
        st.subheader("Feedback Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Submissions", len(df_feedback))
        with col2:
            if 'experimental_bandgap_eV' in df_feedback.columns:
                errors = df_feedback['predicted_bandgap_eV'] - df_feedback['experimental_bandgap_eV']
                mae = np.abs(errors).mean()
                st.metric("User-Reported MAE", f"{mae:.3f} eV")
        with col3:
            if 'experimental_bandgap_eV' in df_feedback.columns:
                rmse = np.sqrt((errors**2).mean())
                st.metric("User-Reported RMSE", f"{rmse:.3f} eV")

    st.markdown("---")
    st.subheader("Admin: Download Data")

    if feedback_file.exists():
        with open(feedback_file, "rb") as f:
            st.download_button(
                label="Download All Feedback (CSV)",
                data=f,
                file_name="user_feedback.csv",
                mime="text/csv"
            )
    else:
        st.info("No feedback data available to download yet.")

# ============================================================================
# PAGE: ABOUT
# ============================================================================

elif page == "About":
    st.markdown('<p class="main-header">About This Tool</p>', unsafe_allow_html=True)
    
    if current_model_key == 'perovskite':
        st.markdown("""
        ### Model: CatBoost Regressor (Tuned)
        
        **Training Data:**
        - **Materials**: ABX₃ Perovskites.
        - **Dataset Size**: ~1400 samples.
        
        **Performance Metrics:**
        - **R² Score**: 0.735
        - **RMSE**: 0.328 eV
        - **MAE**: 0.205 eV
        
        """)
        
    else: # Spinel
        st.markdown("""
        ### Model: CatBoost Regressor (Optimized)
        
        **Training Data:**
        - **Materials**: Cubic, Direct Bandgap Spinels ($AB_2O_4$).
        - **Model trained on**: 
            - Aluminates ($MAl_2O_4$)
            - Ferrites ($MFe_2O_4$)
            - Chromites ($MCr_2O_4$)
            - Cobaltites ($MCo_2O_4$)
        
        **Performance Metrics:**
        - **R² Score**: ~0.62 (Peak ~0.70)
        - **RMSE**: ~0.45 eV
        - **MAE**: ~0.35 eV
        - **Reliability**: Error < 0.20 eV for standard families.

        """)
    
    st.markdown("---")
    
    # Technical details
    with st.expander("Technical Details (Feature Manifest)"):
        st.json(manifest, expanded=False)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("© 2025 Amir Mahboud | RMIT University")
