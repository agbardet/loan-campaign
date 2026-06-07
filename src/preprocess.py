import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["ID", "ZIPCode"])
    df["Experience"] = df["Experience"].clip(lower=0)
    return df.reset_index(drop=True)


def prepare_features(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 1):
    X = df.drop(columns=["Personal_Loan"])
    y = df["Personal_Loan"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
