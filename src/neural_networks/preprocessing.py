from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, OneHotEncoder
from functools import lru_cache
from typing import Any, Callable
import pandas as pd
import inspect

from utils.terminal import getlogger

logger = getlogger()

def encode_label(df : pd.DataFrame, column : str, inplace : bool = True) -> LabelEncoder:
    encoder = LabelEncoder()
    df[f"{column}_encoded" if not inplace else column] = encoder.fit_transform(df[column])

    logger.info(f"Encoded Column '{column}' with LabelEncoder")
    return encoder

def apply_scaler(df : pd.DataFrame, columns : list[str], scaler : MinMaxScaler | StandardScaler | MaxAbsScaler | RobustScaler, inplace : bool = True) -> MinMaxScaler | StandardScaler | MaxAbsScaler | RobustScaler:

    for column in columns:
        df[f"{column}_scaled" if not inplace else column] = scaler.fit_transform(df[[column]])

    logger.info(f"Applied Scalar '{scaler.__class__}' to columns: {columns}")
    return scaler

def apply_onehot_encoder(df: pd.DataFrame, columns: list[str], inplace: bool = True) -> OneHotEncoder:
    encoder = OneHotEncoder(handle_unknown="error")

    for column in columns:
        onehot_encoded_df = pd.DataFrame(encoder.fit_transform(df[[column]]).toarray())
        if not inplace:
            for i, col in enumerate(onehot_encoded_df.columns):
                df[f"{column}_onehot_{i}"] = col
        else:
            for i, col in enumerate(onehot_encoded_df.columns):
                df[column + f"_onehot_{i}"] = col

    logger.info(f"Applied OneHotEncoder to columns: {columns}")
    return encoder

def create_feature(df: pd.DataFrame, feature_name: str, func: Callable[[pd.Series], pd.Series]) -> None:
    """
    Create new features by applying a Lambda function to the input DataFrame.

    Parameters:
        df (pandas.DataFrame): Input DataFrame
        lambda_func (function): Lambda function to apply to the DataFrame
    """
    #@lru_cache(maxsize=1000)
    #def apply_lambda(row : pd.Series):
        #return func(row)

    df[feature_name] = df.apply(lambda row: func(row), axis=1)

    logger.info(f"Created Feature named '{feature_name}': lambda{inspect.getsource(func).strip().split("lambda")[-1]}")