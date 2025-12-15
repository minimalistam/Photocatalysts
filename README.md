# 🔬 ABX₃ & Spinel Bandgap Predictor

A physics-informed machine learning tool for predicting the optical bandgaps of **Perovskite (ABX₃)** and **Spinel (AB₂O₄)** materials. This tool integrates experimental data with physics-based feature engineering to accelerate the discovery of new photocatalysts and optoelectronic materials.

## 🌟 Key Features

*   **Multi-Family Support**: Specialized models for both Perovskite and Spinel oxide families.
*   **Physics-Informed**: Uses features derived from fundamental elemental properties (electronegativity, ionic radius, etc.) rather than just chemical formulas.
*   **Interpretability**: Includes **SHAP (SHapley Additive exPlanations)** waterfall plots to visualize which features drive each prediction.
*   **Batch Processing**: Supports bulk predictions via Excel/CSV upload with automated feature computation.
*   **Uncertainty Quantification**: Provides model uncertainty estimates (RMSE) for every prediction.

## 🧠 Models

The application uses **CatBoost Regressor** models trained on curated experimental datasets.

### Perovskite (ABX₃)
*   **Scope**: Inorganic oxides (e.g., Titanates, Ferrites) and Hybrid Organic-Inorganic Halides.
*   **Performance**:
    *   **R²**: 0.77
    *   **RMSE**: 0.32 eV
    *   **MAE**: 0.19 eV
*   **Training Data**: ~570 experimental samples.

### Spinel (AB₂O₄)
*   **Scope**: Cubic, Direct Bandgap Spinels, primarily Aluminates ($MAl_2O_4$) and Ferrites ($MFe_2O_4$).
*   **Performance**:
    *   **R²**: 0.71
    *   **RMSE**: 0.48 eV
    *   **MAE**: 0.32 eV
*   **Training Data**: ~187 experimental samples.

## 🚀 Usage

### Online Dashboard
Access the interactive dashboard here:
**[Launch App](https://minimalistam-photocatalysts-app.streamlit.app)**

### Local Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/minimalistam/Photocatalysts.git
    cd Photocatalysts
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    streamlit run app.py
    ```

## 📚 Citation

If you use this tool in your research, please cite:

> Mahboud, A. et al. (2025). *Physics-Informed Machine Learning for Bandgap Prediction in Perovskite and Spinel Oxides*. [Journal/Conference Name].

## 📧 Contact

**Amir Mahboud**  
RMIT University  
[amir.mahboud@student.rmit.edu.au](mailto:amir.mahboud@student.rmit.edu.au)
