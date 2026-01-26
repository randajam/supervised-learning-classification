import pandas as pd

def add_mmr_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    eps = 1e-6  # защита от деления на 0

    df["MMR_Acq_ratio"] = (
        df["MMRCurrentAuctionAveragePrice"] /
        (df["MMRAcquisitionAuctionAveragePrice"] + eps)
    )

    df["MMR_Retail_ratio"] = (
        df["MMRCurrentRetailAveragePrice"] /
        (df["MMRAcquisitionRetailAveragePrice"] + eps)
    )

    df["MMR_Acq_diff"] = (
        df["MMRCurrentAuctionAveragePrice"] -
        df["MMRAcquisitionAuctionAveragePrice"]
    )

    df["MMR_Retail_diff"] = (
        df["MMRCurrentRetailAveragePrice"] -
        df["MMRAcquisitionRetailAveragePrice"]
    )

    return df

def add_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    df["Odo_per_year"] = df["VehOdo"] / (df["VehicleAge"] + eps)

    return df

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    # Цена относительно текущей рыночной
    df["Cost_to_MMR_Auction"] = df["VehBCost"] / (df["MMRCurrentAuctionAveragePrice"] + eps)
    df["Cost_to_MMR_Retail"] = df["VehBCost"] / (df["MMRCurrentRetailAveragePrice"] + eps)

    # Разница с рынком
    df["Cost_minus_MMR_Auction"] = df["VehBCost"] - df["MMRCurrentAuctionAveragePrice"]
    df["Cost_minus_MMR_Retail"] = df["VehBCost"] - df["MMRCurrentRetailAveragePrice"]

    return df

def add_frequency_features(train, valid, test, col):
    train = train.copy()
    valid = valid.copy()
    test  = test.copy()

    freq = train[col].value_counts(normalize=True)

    train[f"{col}_freq"] = train[col].map(freq)
    valid[f"{col}_freq"] = valid[col].map(freq)
    test[f"{col}_freq"]  = test[col].map(freq)

    global_freq = freq.mean()

    train[f"{col}_freq"] = train[f"{col}_freq"].fillna(global_freq)
    valid[f"{col}_freq"] = valid[f"{col}_freq"].fillna(global_freq)
    test[f"{col}_freq"]  = test[f"{col}_freq"].fillna(global_freq)

    return train, valid, test

def add_target_encoding(train, valid, test, col, y_train):
    train = train.copy()
    valid = valid.copy()
    test = test.copy()

    # считаем средний bad-rate по категориям ТОЛЬКО на train
    stats = y_train.groupby(train[col]).mean()
    global_mean = y_train.mean()

    train[f"{col}_bad_rate"] = train[col].map(stats).fillna(global_mean)
    valid[f"{col}_bad_rate"] = valid[col].map(stats).fillna(global_mean)
    test[f"{col}_bad_rate"]  = test[col].map(stats).fillna(global_mean)

    return train, valid, test
