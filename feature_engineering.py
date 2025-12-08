"""
Auto-compute physics features from composition.
This module replicates the feature engineering from training.
"""

import pandas as pd
import numpy as np
from pymatgen.core import Element, Composition

# Manual Shannon radii (same as training)
MANUAL_SHANNON_RADII = {
    ('Sn', 2, 6): 1.18,
    ('Sn', 2, 4): 0.93,
    ('Pb', 2, 6): 1.19,
    ('Ge', 2, 6): 0.87,
}

def ionic_radius(el, ox, cn):
    """Get ionic radius (same logic as training)."""
    try:
        el_norm = str(el).strip().capitalize()
        ox_norm = int(ox)
        
        # Check manual database
        key = (el_norm, ox_norm, cn)
        if key in MANUAL_SHANNON_RADII:
            return float(MANUAL_SHANNON_RADII[key])
        
        # Try pymatgen
        e = Element(el_norm)
        R = e.ionic_radii
        if ox_norm in R:
            v = R[ox_norm]
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                if cn in v:
                    return float(v[cn])
                cn_roman = {4: 'IV', 6: 'VI', 8: 'VIII', 12: 'XII'}.get(cn)
                if cn_roman and cn_roman in v:
                    return float(v[cn_roman])
                if v:
                    return float(next(iter(v.values())))
        
        # Fallback to other CNs
        for test_cn in (6, 4, 8, 12):
            k = (el_norm, ox_norm, test_cn)
            if k in MANUAL_SHANNON_RADII:
                return float(MANUAL_SHANNON_RADII[k])
        
        return np.nan
    except:
        return np.nan

def elem_prop(el, attr):
    """Get element property."""
    try:
        e = Element(el)
        if attr == "X": 
            return float(e.X or np.nan)
        if attr == "atomic_mass": 
            return float(e.atomic_mass)
        if attr == "ionization_energy":
            v = e.data.get("Ionization energies", [])
            return float(v[0]) if v else np.nan
        if attr == "electron_affinity":
            return float(e.data.get("Electron affinity") or np.nan)
        return np.nan
    except:
        return np.nan

def get_d_electrons(el):
    """Get d-electrons."""
    try:
        Z = Element(el).Z
        if 21<=Z<=30: return Z-18
        if 39<=Z<=48: return Z-36
        if 57<=Z<=80: return Z-54
        return 0
    except:
        return 0

def compute_physics_features(input_data, elements_data, encoders):
    """
    Compute all physics features from composition.
    
    Args:
        input_data: dict with A/B/X elements and oxidation states
        elements_data: Periodic table data (not used if we use pymatgen)
        encoders: Categorical encoders
        
    Returns:
        DataFrame with one row of features
    """
    
    A_elem = input_data['A_element']
    A_ox = input_data['A_oxidation']
    B_elem = input_data['B_element']
    B_ox = input_data['B_oxidation']
    X_elem = input_data['X_element']
    X_ox = input_data['X_oxidation']
    
    # Compute properties for each site
    features = {}
    
    # A-site features
    A_is_organic = int(A_elem in ['MA', 'FA', 'EA', 'GA'])
    features['A_is_organic'] = A_is_organic
    
    if not A_is_organic:
        features['A_ionic_radius_12CN'] = ionic_radius(A_elem, A_ox, 12)
        features['A_electronegativity'] = elem_prop(A_elem, 'X')
        features['A_atomic_mass'] = elem_prop(A_elem, 'atomic_mass')
        features['A_ionization_energy'] = elem_prop(A_elem, 'ionization_energy')
    else:
        # Organic cation - use approximate values
        features['A_ionic_radius_12CN'] = np.nan
        features['A_electronegativity'] = 2.5 if A_elem == 'MA' else 2.4
        features['A_atomic_mass'] = 120.0 if A_elem == 'FA' else 100.0
        features['A_ionization_energy'] = 7.0
    
    # A-site entropy/variance (set to 0 for single element)
    features['A_entropy'] = 0.0
    features['A_size_variance'] = 0.0
    features['A_EN_var'] = 0.0
    
    # B-site features
    features['B_ionic_radius_6CN'] = ionic_radius(B_elem, B_ox, 6)
    features['B_electronegativity'] = elem_prop(B_elem, 'X')
    features['B_atomic_mass'] = elem_prop(B_elem, 'atomic_mass')
    features['B_ionization_energy'] = elem_prop(B_elem, 'ionization_energy')
    features['B_electron_affinity'] = elem_prop(B_elem, 'electron_affinity')
    features['B_d_electrons'] = get_d_electrons(B_elem)
    features['B_is_transition_metal'] = int(features['B_d_electrons'] > 0)
    features['B_entropy'] = 0.0
    features['B_size_variance'] = 0.0
    features['B_EN_var'] = 0.0
    
    # X-site features
    features['X_ionic_radius_6CN'] = ionic_radius(X_elem, X_ox, 6)
    features['X_electronegativity'] = elem_prop(X_elem, 'X')
    features['X_atomic_mass'] = elem_prop(X_elem, 'atomic_mass')
    features['X_electron_affinity'] = elem_prop(X_elem, 'electron_affinity')
    features['X_entropy'] = 0.0
    features['X_size_variance'] = 0.0
    features['X_EN_var'] = 0.0
    
    # Derived features
    r_A = features['A_ionic_radius_12CN']
    r_B = features['B_ionic_radius_6CN']
    r_X = features['X_ionic_radius_6CN']
    
    if pd.notna(r_A) and pd.notna(r_B) and pd.notna(r_X):
        features['tolerance_factor'] = (r_A + r_X) / (np.sqrt(2) * (r_B + r_X))
    else:
        features['tolerance_factor'] = np.nan
    
    if pd.notna(r_B) and pd.notna(r_X):
        features['octahedral_factor'] = r_B / r_X
    else:
        features['octahedral_factor'] = np.nan
    
    EN_B = features['B_electronegativity']
    EN_X = features['X_electronegativity']
    EN_A = features['A_electronegativity']
    
    features['delta_EN_BX'] = abs(EN_B - EN_X) if pd.notna(EN_B) and pd.notna(EN_X) else np.nan
    features['delta_EN_AX'] = abs(EN_A - EN_X) if pd.notna(EN_A) and pd.notna(EN_X) else np.nan
    features['delta_EN_AB'] = abs(EN_A - EN_B) if pd.notna(EN_A) and pd.notna(EN_B) else np.nan
    
    M_B = features['B_atomic_mass']
    M_X = features['X_atomic_mass']
    
    if pd.notna(M_B) and pd.notna(M_X) and (M_B + M_X) > 0:
        features['reduced_mass_BX'] = (M_B * M_X) / (M_B + M_X)
        features['mass_ratio_BX'] = M_B / M_X
    else:
        features['reduced_mass_BX'] = np.nan
        features['mass_ratio_BX'] = np.nan
    
    if pd.notna(r_X) and pd.notna(r_B) and pd.notna(EN_X) and pd.notna(EN_B) and EN_X > 0 and EN_B > 0:
        features['polarizability_ratio_XB'] = (r_X**3 / EN_X) / (r_B**3 / EN_B)
    else:
        features['polarizability_ratio_XB'] = np.nan
    
    # Synthesis features
    features['synthesis_temperature'] = input_data.get('synthesis_temperature', np.nan)
    features['synthesis_time_hours'] = input_data.get('synthesis_time_hours', np.nan)
    features['annealing_temperature'] = input_data.get('annealing_temperature', np.nan)
    features['annealing_time_hours'] = input_data.get('annealing_time_hours', np.nan)
    features['unit_cell_volume'] = np.nan  # Not provided by user
    
    # Categorical features (encode)
    cat_features = {
        'crystal_structure': input_data.get('crystal_structure', 'Unknown'),
        'sample_form': input_data.get('sample_form', 'Unknown'),
        'synthesis_method': input_data.get('synthesis_method', 'Unknown'),
        'morphology': input_data.get('morphology', 'Unknown'),
        'bandgap_type': input_data.get('bandgap_type', 'Unknown')
    }
    
    for col, value in cat_features.items():
        if col in encoders:
            le = encoders[col]
            # Encode (use -1 for unseen)
            if value in le.classes_:
                features[col] = le.transform([value])[0]
            else:
                features[col] = -1
        else:
            features[col] = 0
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    return df

def validate_composition(input_data, elements_data):
    """
    Validate composition and return warnings.
    
    Returns:
        (is_valid, warnings_list)
    """
    warnings = []
    is_valid = True
    
    A_elem = input_data['A_element']
    B_elem = input_data['B_element']
    X_elem = input_data['X_element']
    
    # Check if elements are valid
    try:
        if A_elem not in ['MA', 'FA', 'EA', 'GA']:
            Element(A_elem)  # Will raise error if invalid
        Element(B_elem)
        Element(X_elem)
    except Exception as e:
        warnings.append(f"Invalid element symbol detected. Please check: {A_elem}, {B_elem}, {X_elem}")
        is_valid = False
        return is_valid, warnings
    
    # Check common combinations (warnings only, not errors)
    if B_elem not in ['Pb', 'Sn', 'Ge', 'Ti', 'Zr', 'Cu', 'Bi', 'Sb']:
        warnings.append(f"B-site {B_elem} is uncommon for perovskites (model trained primarily on Pb, Sn, Ge, Ti)")
    
    if X_elem not in ['I', 'Br', 'Cl', 'F', 'O']:
        warnings.append(f"X-site {X_elem} is uncommon (model trained primarily on I, Br, Cl, F, O)")
    
    return is_valid, warnings


# ============================================================================
# SPINEL FEATURE ENGINEERING
# ============================================================================

def get_t2g_eg_proxy(B_elem):
    """
    Get t2g-eg splitting proxy for B-site cation in Spinel.
    Values derived from training data.
    """
    # Mapping from dataset
    mapping = {
        'Co': 0.689565,
        'Al': 0.709457,
        'Ga': 0.738477, # Inferred from ZnAlGaO4
        'Fe': 0.752500,
        'Cr': 0.793261,
        'Mn': 0.883370,
        'Mg': 0.709457, # MgAl2O4 (A-site Mg, B-site Al) - wait, this function takes B_elem
        # If B is Mg? Unlikely for 2-3 spinel.
    }
    return mapping.get(B_elem, 0.75) # Default to mean if unknown

def compute_spinel_features(input_data, elements_data, encoders):
    """
    Compute features for Spinel (AB2O4) model.
    """
    A_elem = input_data['A_element']
    B_elem = input_data['B_element']
    # X is always O for this model, but we can keep it flexible if needed
    
    features = {}
    
    # 1. Physics Features
    
    # Electronegativity
    EN_A = elem_prop(A_elem, 'X')
    EN_B = elem_prop(B_elem, 'X')
    features['delta_EN_AB'] = abs(EN_A - EN_B) if pd.notna(EN_A) and pd.notna(EN_B) else np.nan
    
    features['A_ionization_energy'] = elem_prop(A_elem, 'ionization_energy')
    features['B_electron_affinity'] = elem_prop(B_elem, 'electron_affinity')
    features['A_d_electrons'] = get_d_electrons(A_elem)
    
    # t2g-eg proxy (Critical feature)
    features['t2g_eg_split_proxy'] = get_t2g_eg_proxy(B_elem)
    
    # 2. User Inputs (Experimental)
    features['lattice_parameter'] = input_data.get('lattice_parameter', np.nan)
    features['crystallite_size'] = input_data.get('crystallite_size', np.nan)
    features['synthesis_temperature'] = input_data.get('synthesis_temperature', np.nan)
    features['synthesis_time_hours'] = input_data.get('synthesis_time_hours', np.nan)
    
    # 3. Categorical Features (Return strings for CatBoost)
    features['synthesis_method'] = str(input_data.get('synthesis_method', 'Unknown'))
    features['morphology'] = str(input_data.get('morphology', 'Unknown'))
    features['bandgap_method'] = str(input_data.get('bandgap_method', 'Unknown'))
            
    # Ensure correct order matching training data
    ordered_cols = [
        "delta_EN_AB",
        "t2g_eg_split_proxy",
        "B_electron_affinity",
        "synthesis_temperature",
        "crystallite_size",
        "lattice_parameter",
        "synthesis_method",
        "A_ionization_energy",
        "bandgap_method",
        "morphology",
        "synthesis_time_hours",
        "A_d_electrons"
    ]
    
    # Create DataFrame with ordered columns
    df = pd.DataFrame([features])
    df = df[ordered_cols]
    
    return df
