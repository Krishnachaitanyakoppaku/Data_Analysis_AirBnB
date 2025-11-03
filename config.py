"""
Configuration file for NYC Airbnb Analysis Project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SRC_DIR = PROJECT_ROOT / "src"

# Data file paths
AIRBNB_DATA_FILE = NOTEBOOKS_DIR / "AB_NYC_2019.csv"

# Analysis parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CLUSTERS = 5

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
STYLE = 'default'

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'max_depth': 10
    },
    'linear_regression': {
        'fit_intercept': True
    },
    'kmeans': {
        'n_clusters': N_CLUSTERS,
        'random_state': RANDOM_STATE,
        'n_init': 10
    }
}

# Output settings
SAVE_FIGURES = True
FIGURE_FORMAT = 'png'
RESULTS_FORMAT = 'csv'

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'