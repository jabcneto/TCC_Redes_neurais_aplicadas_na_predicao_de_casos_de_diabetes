# python
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from evaluation import visualizar_analise_exploratoria_dados
from utils import RANDOM_STATE
from config import LOGGER


"""
Data preprocessing pipeline for classification tasks.

This module implements:
- Dataset loading
- Basic EDA with optional filtering
- Outlier clipping on numeric features using IQR bounds computed on the train set
- Categorical encoding via OneHotEncoder with first-level drop
- Feature scaling via StandardScaler
- Class balancing on the train set with SMOTE

It returns train/validation/test splits ready for model training and the fitted transformers.
"""


def carregar_dados(caminho_arquivo: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV dataset from disk.

    Parameters
    ----------
    caminho_arquivo : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    Optional[pd.DataFrame]
        Loaded DataFrame if successful; otherwise None.

    Side Effects
    ------------
    Logs loading progress and errors through LOGGER.

    Notes
    -----
    The function catches any exception during read and returns None to avoid crashing callers.
    """
    LOGGER.info(f"Carregando dados de {caminho_arquivo}")
    try:
        dataframe = pd.read_csv(caminho_arquivo)
        LOGGER.info(f"Dataset carregado. Formato: {dataframe.shape}")
        return dataframe
    except Exception as e:
        LOGGER.error(f"Erro ao carregar o dataset: {e}")
        return None


def analisar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a lightweight EDA step and optional filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset including all raw features and target.

    Returns
    -------
    pd.DataFrame
        DataFrame possibly filtered, preserving schema except for removed rows.

    Side Effects
    ------------
    Triggers visualization via `visualizar_analise_exploratoria_dados`.

    Notes
    -----
    If the column `gender` contains the category `Other`, those rows are removed before visualization.
    """
    LOGGER.info("Realizando análise exploratória dos dados.")
    if 'gender' in df.columns and 'Other' in df['gender'].unique():
        df = df[df['gender'] != 'Other'].copy()
    visualizar_analise_exploratoria_dados(df)
    return df


def _compute_iqr_bounds(series: pd.Series, fator: float = 1.5) -> Tuple[float, float]:
    """
    Compute lower and upper clipping bounds using the IQR rule.

    Parameters
    ----------
    series : pd.Series
        Numeric series from which to compute Q1, Q3, and IQR.
    fator : float, default=1.5
        Multiplier applied to IQR to expand the whiskers.

    Returns
    -------
    Tuple[float, float]
        Lower and upper bounds for clipping.

    Notes
    -----
    Bounds are computed on the provided series, typically the train subset, to avoid leakage.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - fator * iqr
    upper = q3 + fator * iqr
    return lower, upper


def _apply_bounds(series: pd.Series, bounds: Tuple[float, float]) -> pd.Series:
    """
    Clip a numeric series to provided lower and upper bounds.

    Parameters
    ----------
    series : pd.Series
        Input numeric series to be clipped.
    bounds : Tuple[float, float]
        Lower and upper clipping bounds.

    Returns
    -------
    pd.Series
        Series with values clipped to the specified interval.

    Notes
    -----
    Uses pandas `clip` to efficiently enforce limits while preserving dtype.
    """
    lower, upper = bounds
    return series.clip(lower=lower, upper=upper)


def pre_processar_dados(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series, StandardScaler, OneHotEncoder, List[str]]:
    """
    Build train/validation/test datasets with encoding, scaling, and class balancing.

    Processing Steps
    ----------------
    1. Split into train/validation/test with stratification on the target.
    2. Detect numeric and categorical columns from train dtypes.
    3. Compute IQR bounds on train numeric columns and clip train/val/test consistently.
    4. One-hot encode categorical columns with first level dropped; keep unknowns ignored.
    5. Concatenate numeric and encoded categorical features.
    6. Standardize features using StandardScaler fitted on train.
    7. Apply SMOTE to balance the training set.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset including the target column `diabetes`.
    test_size : float, default=0.2
        Proportion of the dataset assigned to the test split.
    val_size : float, default=0.1
        Proportion of the full dataset assigned to the validation split.
        The function internally converts this into a fraction of the non-test portion.

    Returns
    -------
    Tuple
        (
            X_train_res: np.ndarray,         Balanced and scaled train features,
            X_val_scaled: np.ndarray,        Scaled validation features,
            X_test_scaled: np.ndarray,       Scaled test features,
            y_train_res: pd.Series,          Balanced train targets,
            y_val: pd.Series,                Validation targets,
            y_test: pd.Series,               Test targets,
            scaler: StandardScaler,          Fitted scaler,
            encoder: OneHotEncoder,          Fitted encoder,
            feature_names: List[str]         Feature names after encoding
        )

    Raises
    ------
    KeyError
        If the target column `diabetes` is missing.

    Notes
    -----
    - Numeric vs. categorical detection relies on pandas dtypes.
    - IQR clipping bounds are computed only on the train set to avoid leakage.
    - `OneHotEncoder(handle_unknown='ignore')` prevents failures on unseen categories.
    - SMOTE is applied only to the training split to avoid contaminating validation/test.
    """
    LOGGER.info("Iniciando pré-processamento dos dados.")
    X = df.drop(columns=['diabetes'])
    y = df['diabetes']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1 - test_size),
        random_state=RANDOM_STATE, stratify=y_train_val
    )

    num_cols = X_train.select_dtypes(include=['int64', 'float64', 'float32', 'int32']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    bounds_map = {col: _compute_iqr_bounds(X_train[col]) for col in num_cols}
    X_train[num_cols] = X_train[num_cols].apply(lambda s: _apply_bounds(s, bounds_map[s.name]))
    X_val[num_cols] = X_val[num_cols].apply(lambda s: _apply_bounds(s, bounds_map[s.name]))
    X_test[num_cols] = X_test[num_cols].apply(lambda s: _apply_bounds(s, bounds_map[s.name]))

    try:
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
    except TypeError:
        encoder = OneHotEncoder(drop='first')

    def _to_dense(m):
        """
        Convert sparse matrices to dense ndarrays if applicable.

        Parameters
        ----------
        m : Any
            Input matrix or array.

        Returns
        -------
        Any
            Dense ndarray if input is sparse; otherwise the original object.
        """
        try:
            return m.toarray()
        except Exception:
            return m

    X_train_cat = _to_dense(encoder.fit_transform(X_train[cat_cols])) if cat_cols else None
    X_val_cat = _to_dense(encoder.transform(X_val[cat_cols])) if cat_cols else None
    X_test_cat = _to_dense(encoder.transform(X_test[cat_cols])) if cat_cols else None
    cat_feature_names = list(encoder.get_feature_names_out(cat_cols)) if cat_cols else []

    X_train_num = X_train[num_cols].to_numpy() if num_cols else None
    X_val_num = X_val[num_cols].to_numpy() if num_cols else None
    X_test_num = X_test[num_cols].to_numpy() if num_cols else None

    def _hstack(a, b):
        """
        Horizontally stack two arrays handling None inputs.

        Parameters
        ----------
        a : Optional[np.ndarray]
            First array.
        b : Optional[np.ndarray]
            Second array.

        Returns
        -------
        np.ndarray
            Horizontally stacked array, or the non-None input, or empty array if both are None.
        """
        if a is None and b is None:
            return np.empty((len(X_train), 0))
        if a is None:
            return b
        if b is None:
            return a
        return np.hstack([a, b])

    X_train_final = _hstack(X_train_num, X_train_cat)
    X_val_final = _hstack(X_val_num, X_val_cat)
    X_test_final = _hstack(X_test_num, X_test_cat)

    feature_names = (num_cols if num_cols else []) + list(cat_feature_names)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_val_scaled = scaler.transform(X_val_final)
    X_test_scaled = scaler.transform(X_test_final)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    LOGGER.info(
        f"Divisão dos dados: Treino={X_train_res.shape}, Validação={X_val_scaled.shape}, Teste={X_test_scaled.shape}"
    )

    return X_train_res, X_val_scaled, X_test_scaled, y_train_res, y_val, y_test, scaler, encoder, feature_names