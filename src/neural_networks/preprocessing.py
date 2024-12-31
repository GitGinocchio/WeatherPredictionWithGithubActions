from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from typing import Any
import pandas as pd

def encode_label(df : pd.DataFrame, column : str, inplace : bool = True) -> LabelEncoder:
    encoder = LabelEncoder()
    df[f"{column}_encoded" if not inplace else column] = encoder.fit_transform(df[column])

    return encoder

def apply_scaler(df : pd.DataFrame, columns : list[str], scaler : MinMaxScaler | StandardScaler | MaxAbsScaler | RobustScaler, inplace : bool = True) -> MinMaxScaler | StandardScaler | MaxAbsScaler | RobustScaler:

    for column in columns:
        df[f"{column}_scaled" if not inplace else column] = scaler.fit_transform(df[[column]])

    return scaler
