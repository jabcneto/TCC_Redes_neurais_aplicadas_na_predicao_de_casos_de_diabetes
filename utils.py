import os
import numpy as np
from typing import Dict, Tuple
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve

# --- CONSTANTES GLOBAIS ---
DATASET_PATH = "diabetes_prediction_dataset.csv"
RANDOM_STATE = 42

# --- CONFIGURAÇÕES DE REPRODUTIBILIDADE ---
np.random.seed(RANDOM_STATE)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- MAPEAMENTO DE COLUNAS ---
MAPEAMENTO_COLUNAS_PT = {
    'gender': 'gênero',
    'age': 'idade',
    'hypertension': 'hipertensão',
    'heart_disease': 'doença cardíaca',
    'smoking_history': 'histórico de tabagismo',
    'bmi': 'IMC',
    'HbA1c_level': 'nível de HbA1c',
    'blood_glucose_level': 'nível de glicose no sangue',
    'diabetes': 'diabetes'
}


def compute_class_weights_from_labels(y) -> Dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return {int(k): float(v) for k, v in zip(classes, weights)}


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                           objective: str = 'f1',
                           min_recall: float | None = None,
                           min_precision: float | None = None) -> Tuple[float, dict]:
    """
    Encontra o threshold ótimo baseado na curva Precision-Recall.

    Parâmetros:
    - objective: 'f1' para maximizar F1; 'precision' para maximizar precisão; 'recall' para maximizar recall.
    - min_recall/min_precision: restrições opcionais (por exemplo, garantir recall >= 0.7 enquanto maximiza precisão).

    Retorna: (threshold, {'precision': p, 'recall': r, 'f1': f1})
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Alinhar arrays: thresholds tem len N-1
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if len(thresholds) == 0:
        return 0.5, {'precision': float('nan'), 'recall': float('nan'), 'f1': float('nan')}

    if objective == 'precision':
        mask = np.ones_like(precisions, dtype=bool)
        if min_recall is not None:
            mask &= recalls >= min_recall
        if not np.any(mask):
            mask = slice(None)
        scores = precisions.copy()
        best_idx = np.argmax(scores[mask])
        # Map back to original indices if masked
        idxs = np.where(mask)[0]
        best_idx = idxs[best_idx] if isinstance(mask, np.ndarray) else best_idx
    elif objective == 'recall':
        mask = np.ones_like(recalls, dtype=bool)
        if min_precision is not None:
            mask &= precisions >= min_precision
        if not np.any(mask):
            mask = slice(None)
        scores = recalls.copy()
        best_idx = np.argmax(scores[mask])
        idxs = np.where(mask)[0]
        best_idx = idxs[best_idx] if isinstance(mask, np.ndarray) else best_idx
    else:
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
        if min_recall is not None:
            f1_scores = np.where(recalls >= min_recall, f1_scores, -np.inf)
        if min_precision is not None:
            f1_scores = np.where(precisions >= min_precision, f1_scores, -np.inf)
        best_idx = int(np.nanargmax(f1_scores))

    thr = float(thresholds[best_idx])
    p = float(precisions[best_idx])
    r = float(recalls[best_idx])
    f1 = float(2 * p * r / (p + r + 1e-12))
    return thr, {'precision': p, 'recall': r, 'f1': f1}
