# dataset.py
#
# Loads the Breast Cancer Wisconsin Diagnostic (WDBC) dataset.
# Replaces the old fake patient dataset.
#
# This integrates perfectly with model.py, encrypt.py, demo.py, demo_helper.py.

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Path to the WDBC .data file (uploaded by you)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wdbc.data")

def load_data():
    """
    Loads the WDBC dataset from the local wdbc.data file.
    Converts labels to 0/1 and returns train/test splits.
    """

    # WDBC has no header row, so we define the columns manually:
    colnames = [
        "ID", "Diagnosis",
        # 30 numeric features:
        "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
        "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
        "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
        "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
        "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
        "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
    ]

    # Load CSV with no header
    df = pd.read_csv(DATA_PATH, header=None, names=colnames)

    # Convert labels: M = 1 (malignant), B = 0 (benign)
    df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

    # Extract features (X) and labels (y)
    X = df.iloc[:, 2:].values      # all 30 numeric features
    y = df["Diagnosis"].values     # 0/1 labels

    # 🔥 Standardize features (helps logistic regression + CKKS precision)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split (same shape as old code expects)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
