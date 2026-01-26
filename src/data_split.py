import pandas as pd
from typing import Optional, Tuple

def date_split(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    date_col: str,
    train_size: float,
    val_size: float
) -> tuple:
    """
    Splits dataset by date into train/val/test.
    Supports both forms:
    
    1) X, y supplied separately:
       (X_train, X_val, X_test, y_train, y_val, y_test)

    2) Only X supplied, with y=None:
       (df_train, df_val, df_test)
    """

    if date_col not in X.columns:
        raise ValueError(f"{date_col} is not a valid column name")

    X = X.copy()

    X[date_col] = pd.to_datetime(X[date_col])
    X = X.sort_values(by=date_col)

    if y is not None:
        y = y.reindex(X.index).reset_index(drop=True)

    X = X.reset_index(drop=True)

    train_idx = int(len(X) * train_size)
    val_idx = train_idx + int(len(X) * val_size)

    while train_idx < len(X)-1 and X[date_col][train_idx - 1] == X[date_col][train_idx]:
        train_idx += 1

    while val_idx < len(X)-1 and X[date_col][val_idx - 1] == X[date_col][val_idx]:
        val_idx += 1

    X_train = X[:train_idx].reset_index(drop=True)
    X_val = X[train_idx:val_idx].reset_index(drop=True)
    X_test = X[val_idx:].reset_index(drop=True)

    if y is None:
        return X_train, X_val, X_test

    return (
        X_train,
        X_val,
        X_test,
        y[:train_idx].reset_index(drop=True),
        y[train_idx:val_idx].reset_index(drop=True),
        y[val_idx:].reset_index(drop=True)
    )
