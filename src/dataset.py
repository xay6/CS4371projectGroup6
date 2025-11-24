# dataset.py
#
# Makes a fake patient dataset.
# Just random numbers for school demo.

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "fake_patients.csv")


def make_fake_data(num=200):
    """Makes random patient data + labels (totally fake)."""

    rng = np.random.default_rng(42)

    age = rng.integers(25, 80, size=num)
    chol = rng.normal(200, 30, size=num)
    biom = rng.normal(1.0, 0.5, size=num)
    risk = rng.normal(0.0, 1.0, size=num)

    # Fake rule to generate cancer label
    z = (0.03 * (age - 50)
         + 0.02 * (chol - 200)
         + 1.2 * (biom - 1.0)
         + 0.5 * risk)
    p = 1 / (1 + np.exp(-z))
    label = (p > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age,
        "cholesterol": chol,
        "biomarker": biom,
        "risk_score": risk,
        "cancer_type": label
    })

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print("Saved fake data to:", DATA_PATH)

    return df


def load_data():
    """Loads dataset and makes train/test split."""
    if not os.path.exists(DATA_PATH):
        make_fake_data()

    df = pd.read_csv(DATA_PATH)

    X = df[["age", "cholesterol", "biomarker", "risk_score"]].values
    y = df["cancer_type"].values

    return train_test_split(X, y, test_size=0.3, random_state=42)