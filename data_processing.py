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


def carregar_dados(caminho_arquivo: str) -> Optional[pd.DataFrame]:
    LOGGER.info(f"Carregando dados de {caminho_arquivo}")
    try:
        df = pd.read_csv(caminho_arquivo)
        LOGGER.info(f"Dataset carregado. Formato: {df.shape}")
        return df
    except Exception as e:
        LOGGER.error(f"Erro ao carregar o dataset: {e}")
        return None


def analisar_dados(df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Realizando análise exploratória dos dados.")
    if 'gender' in df.columns and 'Other' in df['gender'].unique():
        df = df[df['gender'] != 'Other'].copy()
    visualizar_analise_exploratoria_dados(df)
    return df


def _compute_iqr_bounds(series: pd.Series, fator: float = 1.5) -> Tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - fator * iqr, q3 + fator * iqr


def pre_processar_dados(
    df: pd.DataFrame,
    test_size: float = 0.35,
    val_size: float = 0.05,
    balance_strategy: str = "smote",
    sampling_strategy: float = 0.4,
    k_neighbors: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series, StandardScaler, OneHotEncoder, List[str]]:
    LOGGER.info("Iniciando pré-processamento dos dados.")
    X = df.drop(columns=['diabetes'])
    y = df['diabetes']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1 - test_size), random_state=RANDOM_STATE, stratify=y_train_val
    )

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    if num_cols:
        # Compute per-column IQR bounds on train and reuse them to avoid leakage
        bounds = pd.DataFrame({col: _compute_iqr_bounds(X_train[col]) for col in num_cols}, index=['low', 'up']).T
        lower = bounds['low']  # lower bounds per column
        upper = bounds['up']   # upper bounds per column
        # Vectorized clipping per column for train/val/test using the same bounds
        X_train[num_cols] = X_train[num_cols].clip(lower=lower, upper=upper, axis=1)
        X_val[num_cols] = X_val[num_cols].clip(lower=lower, upper=upper, axis=1)
        X_test[num_cols] = X_test[num_cols].clip(lower=lower, upper=upper, axis=1)

    # One-hot encode categorical columns (drop first; ignore unknowns)
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    # Fit on train; transform val/test; if no categorical features, return empty arrays
    X_train_cat = encoder.fit_transform(X_train[cat_cols]) if cat_cols else np.empty((len(X_train), 0))
    X_val_cat = encoder.transform(X_val[cat_cols]) if cat_cols else np.empty((len(X_val), 0))
    X_test_cat = encoder.transform(X_test[cat_cols]) if cat_cols else np.empty((len(X_test), 0))
    cat_feature_names = list(encoder.get_feature_names_out(cat_cols)) if cat_cols else []

    # Convert numeric DataFrames to numpy arrays; if none, use empty arrays (n, 0)
    X_train_num = X_train[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(X_train), 0))
    X_val_num = X_val[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(X_val), 0))
    X_test_num = X_test[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(X_test), 0))

    # Concatenate numeric and categorical blocks
    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_val_final = np.hstack([X_val_num, X_val_cat])
    X_test_final = np.hstack([X_test_num, X_test_cat])

    # Final feature names after encoding
    feature_names = (num_cols if num_cols else []) + cat_feature_names

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_val_scaled = scaler.transform(X_val_final)
    X_test_scaled = scaler.transform(X_test_final)

    X_train_res = X_train_scaled
    y_train_res = y_train

    if balance_strategy and balance_strategy.lower() == 'smote':
        class_counts = y_train.value_counts()
        ratio = class_counts.min() / class_counts.max()
        if ratio < 0.65:
            sampler = SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)
            X_train_res, y_train_res = sampler.fit_resample(X_train_scaled, y_train)

    LOGGER.info(
        f"Divisão dos dados: Treino={X_train_res.shape}, Validação={X_val_scaled.shape}, Teste={X_test_scaled.shape}"
    )

    return X_train_res, X_val_scaled, X_test_scaled, y_train_res, y_val, y_test, scaler, encoder, feature_names
