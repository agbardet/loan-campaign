import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Experience dropped: r=0.99 with Age — redundant feature
    df = df.drop(columns=["ID", "Experience"])
    # Truncate ZIPCode to 2-digit prefix then one-hot encode
    df["ZIPCode"] = df["ZIPCode"].astype(str).str[:2]
    # Education: integer → string labels then one-hot encode
    df["Education"] = df["Education"].replace({1: "Undergrad", 2: "Graduate", 3: "Advanced_Professional"})
    df = pd.get_dummies(df, columns=["ZIPCode", "Education"], dtype=int, drop_first=True)
    return df.reset_index(drop=True)


def prepare_features(df: pd.DataFrame, test_size: float = 0.30, random_state: int = 1):
    X = df.drop(columns=["Personal_Loan"])
    y = df["Personal_Loan"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
