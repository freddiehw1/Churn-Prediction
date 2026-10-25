"""Load, validate and feature-engineer the public bank churn dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = 'https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Bank%20Churn%20Modelling.csv'
DATA_PATH = Path(__file__).resolve().parent / 'data' / 'bank_churn.csv'

# Raw model inputs. CustomerId and Surname are identifiers, not features, so they are never used.
FEATURES = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
TARGET = 'Exited'
CATEGORICAL = ['Geography', 'Gender']
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

# Columns created by add_features(). Each one comes from a pattern found in the exploratory analysis.
ENGINEERED_NUMERIC = ['ZeroBalance', 'Products3Plus', 'Products2', 'AgeSquared', 'InactiveOver40']

COLUMN_RENAMES = {
    'Num Of Products': 'NumOfProducts',
    'Has Credit Card': 'HasCrCard',
    'Is Active Member': 'IsActiveMember',
    'Estimated Salary': 'EstimatedSalary',
    'Churn': 'Exited',
}


def load_data():
    """Read the cached CSV (downloading it on first run), standardise column names and validate."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        try:
            df = pd.read_csv(DATA_URL)
        except Exception as exc:
            raise RuntimeError('Could not download the dataset. Download it manually from the source URL '
                               'in README.md and save it as data/bank_churn.csv.') from exc
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    df = df.rename(columns=COLUMN_RENAMES)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f'Dataset is missing required columns: {sorted(missing)}')
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError('Target Exited must be binary 0/1')
    if 'CustomerId' in df.columns and df['CustomerId'].duplicated().any():
        raise ValueError('Duplicate CustomerId rows found')
    return df


def add_features(X):
    """Add engineered features. Uses only the row's own inputs (never the target), so it is leakage-safe
    and can sit inside the sklearn pipeline."""
    X = X.copy()
    X['ZeroBalance'] = (X['Balance'] == 0).astype(int)
    # Churn by product count is not monotonic: 1 product ~28%, 2 products ~8%, 3+ products ~85%.
    # A single linear coefficient cannot represent that U-shape, so give the linear model flags.
    X['Products3Plus'] = (X['NumOfProducts'] >= 3).astype(int)
    X['Products2'] = (X['NumOfProducts'] == 2).astype(int)
    # Churn rises with age to a peak in the 50s and then falls: the squared term lets a linear model bend.
    X['AgeSquared'] = np.square(X['Age'].astype(float))
    # Inactive customers over 40 churn far more than active customers of the same age.
    X['InactiveOver40'] = ((X['IsActiveMember'] == 0) & (X['Age'] > 40)).astype(int)
    return X
