"""
Element data and dropdown options for the app.
"""

# A-site options (common perovskite A-site cations)
A_SITE_OPTIONS = [
    'Cs', 'Rb', 'K', 'Na',  # Inorganic
    'MA', 'FA', 'EA', 'GA',  # Organic (methylammonium, formamidinium, etc.)
    'Ca', 'Sr', 'Ba',  # Alkaline earth
    'La', 'Ce', 'Pr', 'Nd',  # Lanthanides
]

# B-site options (common perovskite B-site cations)
B_SITE_OPTIONS = [
    'Pb', 'Sn', 'Ge',  # Group 14
    'Ti', 'Zr', 'Hf',  # Group 4
    'V', 'Nb', 'Ta',  # Group 5
    'Cr', 'Mo', 'W',  # Group 6
    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',  # 3d transition metals
    'Bi', 'Sb',  # Group 15
    'Al', 'Ga', 'In',  # Group 13
]

# X-site options (halides + oxygen)
X_SITE_OPTIONS = [
    'I', 'Br', 'Cl', 'F',  # Halides
    'O', 'S', 'Se', 'Te',  # Chalcogenides
    'N',  # Nitride
]

# Placeholder for element properties
# (pymatgen will provide most of these, this is just for reference)
ELEMENTS_DATA = {
    'MA': {  # Methylammonium (organic)
        'electronegativity': 2.5,
        'atomic_mass': 31.06,
        'is_organic': True
    },
    'FA': {  # Formamidinium (organic)
        'electronegativity': 2.4,
        'atomic_mass': 45.08,
        'is_organic': True
    },
    'EA': {  # Ethylammonium (organic)
        'electronegativity': 2.5,
        'atomic_mass': 45.08,
        'is_organic': True
    },
    'GA': {  # Guanidinium (organic)
        'electronegativity': 2.5,
        'atomic_mass': 59.07,
        'is_organic': True
    }
}


