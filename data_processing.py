import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from evaluation import visualizar_analise_exploratoria_dados
from utils import RANDOM_STATE
from config import LOGGER


def carregar_dados(caminho_arquivo):
    LOGGER.info(f"Carregando dados de {caminho_arquivo}")
    try:
        dataframe = pd.read_csv(caminho_arquivo)
        LOGGER.info(f"Dataset carregado. Formato: {dataframe.shape}")
        return dataframe
    except Exception as e:
        LOGGER.error(f"Erro ao carregar o dataset: {e}")
        return None


def analisar_dados(df):
    LOGGER.info("Realizando análise exploratória dos dados.")
    if 'gender' in df.columns and 'Other' in df['gender'].unique():
        df = df[df['gender'] != 'Other'].copy()
    visualizar_analise_exploratoria_dados(df)
    return df


def _compute_iqr_bounds(series: pd.Series, fator: float = 1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - fator * iqr
    upper = q3 + fator * iqr
    return lower, upper


def _apply_bounds(series: pd.Series, bounds):
    lower, upper = bounds
    return series.clip(lower=lower, upper=upper)


def pre_processar_dados(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
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

    import numpy as np
    def _hstack(a, b):
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
