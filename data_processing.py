# python
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from evaluation import visualizar_analise_exploratoria_dados
from utils import RANDOM_STATE
from config import LOGGER, RESULTS_DIR
import os


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


class CombinedCategoricalEncoder:
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols
        self.ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        try:
            self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            self.ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self._feature_names = None

    def fit(self, df_cat):
        X_ord = self.ordinal.fit_transform(df_cat)
        self.ohe.fit(X_ord)
        cats = self.ohe.categories_
        names = []
        for col, cat_vals in zip(self.cat_cols, cats):
            for v in cat_vals:
                names.append(f"{col}_{int(v)}")
        self._feature_names = names
        return self

    def transform(self, df_cat):
        X_ord = self.ordinal.transform(df_cat)
        return self.ohe.transform(X_ord)

    def get_feature_names_out(self, input_features=None):
        return np.array(self._feature_names or [])


def _plot_resampled_distribution(y, name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    counts = pd.Series(y).value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette='viridis')
    plt.title(f'Class distribution - {name}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    out_path = os.path.join(RESULTS_DIR, 'graficos', 'distribuicao', f'dist_{name}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    LOGGER.info(f"Gráfico de distribuição salvo: {out_path}")


def _plot_before_after_distribution(y_before, y_after, name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    counts_before = pd.Series(y_before).value_counts().sort_index()
    counts_after = pd.Series(y_after).value_counts().sort_index()
    sns.barplot(ax=axes[0], x=counts_before.index.astype(str), y=counts_before.values, palette='viridis')
    axes[0].set_title('Before')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    sns.barplot(ax=axes[1], x=counts_after.index.astype(str), y=counts_after.values, palette='viridis')
    axes[1].set_title('After')
    axes[1].set_xlabel('Class')
    out_path = os.path.join(RESULTS_DIR, 'graficos', 'distribuicao', f'dist_before_after_{name}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    LOGGER.info(f"Gráfico de distribuição antes/depois salvo: {out_path}")


def pre_processar_dados(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    balance_strategy: str = "smote",
    sampling_strategy: float = 0.7,
    k_neighbors: int = 5,
    auto_balance: bool = False
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
    balance_strategy : str, default="smote"
        Resampling strategy for class balancing. Options: "smote", "adasyn", "smote_tomek", "smote_enn", "ros", "rus", "none".
    sampling_strategy : float, default=0.7
        Target ratio for the minority class after resampling.
    k_neighbors : int, default=5
        Number of neighbors to use for KNN-based resampling methods (SMOTE, ADASYN).
    auto_balance : bool, default=False
        If True, automatically selects the best balancing strategy based on validation precision.

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

    class_counts = y_train.value_counts()
    LOGGER.info(f"Distribuição original das classes no treino: {class_counts.to_dict()}")

    X_train_res = X_train_scaled
    y_train_res = y_train

    if auto_balance:
        LOGGER.info("Auto-balance ativado: testando estratégias para maximizar precisão de validação.")
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_score
        from utils import find_optimal_threshold

        def _score_strategy(x_tr, y_tr, x_v, y_v):
            try:
                clf = LogisticRegression(max_iter=1000, solver='liblinear')
                clf.fit(x_tr, y_tr)
                p = clf.predict_proba(x_v)[:, 1]
                thr, _ = find_optimal_threshold(y_v.values if hasattr(y_v, 'values') else y_v, p, objective='precision', min_recall=0.5)
                y_hat = (p > thr).astype(int)
                return precision_score(y_v, y_hat)
            except Exception:
                return 0.0

        candidates = [
            ("none", None),
            ("smote", 0.3),
            ("smote", 0.5),
            ("smote_tomek", 0.5),
            ("smote_enn", 0.5),
            ("ros", 0.5),
            ("rus", 0.5),
        ]
        best_name = "none"
        best_ss = None
        best_score = -1.0

        for name, ss in candidates:
            try:
                if name == "none":
                    x_bal, y_bal = X_train_scaled, y_train
                else:
                    if name == "smote":
                        sampler = SMOTE(random_state=RANDOM_STATE, sampling_strategy=ss, k_neighbors=k_neighbors)
                    elif name == "adasyn":
                        sampler = ADASYN(random_state=RANDOM_STATE, sampling_strategy=ss, n_neighbors=k_neighbors)
                    elif name == "smote_tomek":
                        sampler = SMOTETomek(random_state=RANDOM_STATE, sampling_strategy=ss)
                    elif name == "smote_enn":
                        sampler = SMOTEENN(random_state=RANDOM_STATE, sampling_strategy=ss)
                    elif name == "ros":
                        sampler = RandomOverSampler(random_state=RANDOM_STATE, sampling_strategy=ss)
                    elif name == "rus":
                        sampler = RandomUnderSampler(random_state=RANDOM_STATE, sampling_strategy=ss)
                    else:
                        sampler = SMOTE(random_state=RANDOM_STATE, sampling_strategy=ss, k_neighbors=k_neighbors)
                    x_bal, y_bal = sampler.fit_resample(X_train_scaled, y_train)
                score = _score_strategy(x_bal, y_bal, X_val_scaled, y_val)
                LOGGER.info(f"Estratégia testada: {name} (ss={ss}) -> val_precision={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_ss = ss
            except Exception as e:
                LOGGER.warning(f"Falha ao testar estratégia {name} (ss={ss}): {e}")

        if best_name == "none":
            LOGGER.info("Auto-balance escolheu: none (sem reamostragem)")
            X_train_res, y_train_res = X_train_scaled, y_train
        else:
            LOGGER.info(f"Auto-balance escolheu: {best_name} com sampling_strategy={best_ss}")
            try:
                if best_name == "smote":
                    sampler = SMOTE(random_state=RANDOM_STATE, sampling_strategy=best_ss, k_neighbors=k_neighbors)
                elif best_name == "adasyn":
                    sampler = ADASYN(random_state=RANDOM_STATE, sampling_strategy=best_ss, n_neighbors=k_neighbors)
                elif best_name == "smote_tomek":
                    sampler = SMOTETomek(random_state=RANDOM_STATE, sampling_strategy=best_ss)
                elif best_name == "smote_enn":
                    sampler = SMOTEENN(random_state=RANDOM_STATE, sampling_strategy=best_ss)
                elif best_name == "ros":
                    sampler = RandomOverSampler(random_state=RANDOM_STATE, sampling_strategy=best_ss)
                elif best_name == "rus":
                    sampler = RandomUnderSampler(random_state=RANDOM_STATE, sampling_strategy=best_ss)
                else:
                    sampler = SMOTE(random_state=RANDOM_STATE, sampling_strategy=best_ss, k_neighbors=k_neighbors)
                X_train_res, y_train_res = sampler.fit_resample(X_train_scaled, y_train)
                LOGGER.info(f"Nova distribuição após auto-balance: {pd.Series(y_train_res).value_counts().to_dict()}")
                _plot_resampled_distribution(y_train_res, f"train_post_{best_name}_{str(best_ss).replace('.', '')}")
                _plot_before_after_distribution(y_train, y_train_res, f"train_{best_name}_{str(best_ss).replace('.', '')}")
            except Exception as e:
                LOGGER.warning(f"Falha ao aplicar estratégia escolhida. Mantendo dados originais. Erro: {e}")
                X_train_res, y_train_res = X_train_scaled, y_train
    else:
        minority_class = class_counts.min()
        majority_class = class_counts.max()
        ratio = minority_class / majority_class
        apply_balance = balance_strategy.lower() != "none" and ratio < 0.95
        if apply_balance:
            bs = balance_strategy.lower()
            try:
                if bs == "smotenc":
                    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
                    enc = CombinedCategoricalEncoder(cat_cols)
                    enc.fit(X_train[cat_cols])
                    X_train_num_arr = X_train[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(X_train), 0))
                    X_train_cat_ord = enc.ordinal.transform(X_train[cat_cols]) if cat_cols else np.empty((len(X_train), 0))
                    X_train_for_smote = np.hstack([X_train_num_arr, X_train_cat_ord])
                    sampler = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)
                    X_resampled, y_resampled = sampler.fit_resample(X_train_for_smote, y_train)
                    X_train_num_res = X_resampled[:, :len(num_cols)] if len(num_cols) > 0 else np.empty((len(X_resampled), 0))
                    X_train_cat_res_ord = X_resampled[:, len(num_cols):] if len(cat_cols) > 0 else np.empty((len(X_resampled), 0))
                    enc.ohe.fit(X_train_cat_res_ord)
                    X_train_cat_res_ohe = enc.ohe.transform(X_train_cat_res_ord)
                    X_val_cat_ohe = enc.transform(X_val[cat_cols]) if cat_cols else np.empty((len(X_val), 0))
                    X_test_cat_ohe = enc.transform(X_test[cat_cols]) if cat_cols else np.empty((len(X_test), 0))
                    X_train_final = np.hstack([X_train_num_res, X_train_cat_res_ohe])
                    X_val_final = np.hstack([X_val[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(X_val), 0), dtype=float), X_val_cat_ohe])
                    X_test_final = np.hstack([X_test[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(X_test), 0), dtype=float), X_test_cat_ohe])
                    feature_names = (num_cols if num_cols else []) + list(enc.get_feature_names_out())
                    scaler = StandardScaler()
                    X_train_res = scaler.fit_transform(X_train_final)
                    X_val_scaled = scaler.transform(X_val_final)
                    X_test_scaled = scaler.transform(X_test_final)
                    y_train_res = pd.Series(y_resampled)
                    LOGGER.info(f"Balanceamento 'smotenc' aplicado com sampling_strategy={sampling_strategy}. Nova distribuição: {y_train_res.value_counts().to_dict()}")
                    _plot_resampled_distribution(y_train_res, f"train_post_smotenc_{str(sampling_strategy).replace('.', '')}")
                    _plot_before_after_distribution(y_train, y_train_res, f"train_smotenc_{str(sampling_strategy).replace('.', '')}")
                    LOGGER.info(
                        f"Divisão dos dados: Treino={X_train_res.shape}, Validação={X_val_scaled.shape}, Teste={X_test_scaled.shape}"
                    )
                    return X_train_res, X_val_scaled, X_test_scaled, y_train_res, y_val, y_test, scaler, enc, feature_names
                elif bs == "smote":
                    sampler = SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)
                elif bs == "adasyn":
                    sampler = ADASYN(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy, n_neighbors=k_neighbors)
                elif bs == "smote_tomek":
                    sampler = SMOTETomek(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)
                elif bs == "smote_enn":
                    sampler = SMOTEENN(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)
                elif bs == "ros":
                    sampler = RandomOverSampler(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)
                elif bs == "rus":
                    sampler = RandomUnderSampler(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)
                else:
                    sampler = SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)

                X_train_res, y_train_res = sampler.fit_resample(X_train_scaled, y_train)
                LOGGER.info(f"Balanceamento '{bs}' aplicado com sampling_strategy={sampling_strategy}. Nova distribuição: {pd.Series(y_train_res).value_counts().to_dict()}")
                _plot_resampled_distribution(y_train_res, f"train_post_{bs}_{str(sampling_strategy).replace('.', '')}")
                _plot_before_after_distribution(y_train, y_train_res, f"train_{bs}_{str(sampling_strategy).replace('.', '')}")
            except Exception as e:
                LOGGER.warning(f"Falha ao aplicar '{balance_strategy}'. Continuando sem reamostragem. Erro: {e}")
                X_train_res = X_train_scaled
                y_train_res = y_train
        else:
            LOGGER.info("Balanceamento não aplicado (estratégia 'none' ou classes já próximas do equilíbrio).")

    LOGGER.info(
        f"Divisão dos dados: Treino={X_train_res.shape}, Validação={X_val_scaled.shape}, Teste={X_test_scaled.shape}"
    )

    return X_train_res, X_val_scaled, X_test_scaled, y_train_res, y_val, y_test, scaler, encoder, feature_names
