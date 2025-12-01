import os



import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import pickle
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import RESULTS_DIR, LOGGER


def criar_callbacks_pt(nome_modelo, paciencia=25, monitor='val_pr_auc'):
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
    log_dir = os.path.join(RESULTS_DIR, "logs", f"{nome_modelo}_{time.strftime('%Y%m%d-%H%M%S')}")
    best_model_path = os.path.join(RESULTS_DIR, "modelos", f"{nome_modelo}_best.keras")

    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "history"), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    monitor_lower = (monitor or '').lower()
    mode = 'max' if ('acc' in monitor_lower or 'auc' in monitor_lower or 'f1' in monitor_lower or 'precision' in monitor_lower or 'recall' in monitor_lower) else 'min'

    LOGGER.info(f"Callback monitorando '{monitor}' no modo '{mode}'.")

    return [
        EarlyStopping(
            monitor=monitor,
            patience=paciencia,
            verbose=1,
            mode=mode,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            save_best_only=True,
            monitor=monitor,
            mode=mode,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(1, paciencia // 2),
            min_lr=1e-7,
            mode=mode,
            verbose=1
        ),
        CSVLogger(filename=os.path.join(RESULTS_DIR, "history", f"{nome_modelo}_history.csv")),
        TensorBoard(log_dir=log_dir)
    ]


def _save_history_summary(history, nome_modelo: str):
    try:
        hist = history.history if hasattr(history, 'history') else (history or {})
        metrics_to_summarize = [
            'loss', 'accuracy', 'precision', 'recall', 'auc', 'pr_auc',
            'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_auc', 'val_pr_auc'
        ]
        rows = []
        for m in metrics_to_summarize:
            values = hist.get(m)
            if values is None or len(values) == 0:
                continue
            arr = np.array(values, dtype=float)
            rows.append({
                'metric': m,
                'mean': float(np.nanmean(arr)),
                'std': float(np.nanstd(arr)),
                'min': float(np.nanmin(arr)),
                'max': float(np.nanmax(arr)),
            })
        if not rows:
            LOGGER.warning("Histórico vazio ou sem métricas esperadas para resumo.")
            return None
        df = pd.DataFrame(rows)
        save_dir = os.path.join(RESULTS_DIR, 'history')
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{nome_modelo}_history_summary.csv")
        df.to_csv(out_path, index=False)
        LOGGER.info(f"Resumo de histórico salvo em: {out_path}")
        return out_path
    except Exception as e:
        LOGGER.warning(f"Falha ao salvar resumo do histórico: {e}")
        return None


def treinar_modelo_keras_pt(model, x_train, y_train, x_val, y_val, nome_modelo, epochs=150, batch_size=64, class_weight=None):
    LOGGER.info(f"Iniciando treinamento do modelo: {nome_modelo}")
    callbacks = criar_callbacks_pt(nome_modelo, monitor='val_pr_auc')

    if "cnn" in nome_modelo.lower() or "hibrido" in nome_modelo.lower():
        try:
            input_shape = model.input_shape
            expects_rank = len(input_shape) if input_shape is not None else 0
        except Exception:
            expects_rank = 0
        if expects_rank >= 3 and len(x_train.shape) == 2:
            x_train = np.expand_dims(x_train, axis=-1)
            x_val = np.expand_dims(x_val, axis=-1)

    if class_weight is None:
        try:
            from utils import compute_class_weights_from_labels
            class_weight = compute_class_weights_from_labels(y_train)
            LOGGER.info(f"class_weight aplicado: {class_weight}")
        except Exception as e:
            LOGGER.warning(f"Falha ao calcular class_weight: {e}")
            class_weight = None

    fit_kwargs = dict(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight

    history = model.fit(**fit_kwargs)
    os.makedirs(os.path.join(RESULTS_DIR, "modelos"), exist_ok=True)
    model.save(os.path.join(RESULTS_DIR, "modelos", f"{nome_modelo}_final.keras"))
    _save_history_summary(history, nome_modelo)
    return model, history


def treinar_modelos_classicos_pt(models, x_train, y_train):
    trained_models = {}

    LOGGER.info("Iniciando treinamento dos modelos clássicos...")
    try:
        from utils import compute_class_weights_from_labels
        class_weights = compute_class_weights_from_labels(y_train)
        sample_weight = np.array([class_weights[int(y)] for y in y_train])
    except Exception as e:
        LOGGER.warning(f"Falha ao calcular class/sample weights: {e}")
        sample_weight = None

    for name, model in tqdm(models.items(), desc="Treinando modelos clássicos"):
        if hasattr(model, 'verbose'):
            model.verbose = 0
        if hasattr(model, 'verbosity'):
            model.verbosity = 0

        if sample_weight is not None:
            try:
                model.fit(x_train, y_train, sample_weight=sample_weight)
            except TypeError:
                model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train)

        os.makedirs(os.path.join(RESULTS_DIR, "modelos"), exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "modelos", f"{name.replace(' ', '_').lower()}.pkl"), 'wb') as f:
            data = pickle.dumps(model)
            f.write(data)

        trained_models[name] = model

    LOGGER.info("Treinamento dos modelos clássicos concluído.")
    return trained_models


def summarize_history_csv(nome_modelo: str) -> str | None:
    try:
        csv_path = os.path.join(RESULTS_DIR, 'history', f"{nome_modelo}_history.csv")
        if not os.path.exists(csv_path):
            LOGGER.warning(f"CSV de histórico não encontrado: {csv_path}")
            return None
        df = pd.read_csv(csv_path)
        metrics_to_summarize = [
            'loss', 'accuracy', 'precision', 'recall', 'auc', 'pr_auc',
            'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_auc', 'val_pr_auc'
        ]
        rows = []
        for m in metrics_to_summarize:
            if m not in df.columns:
                continue
            s = pd.to_numeric(df[m], errors='coerce')
            rows.append({
                'metric': m,
                'mean': float(s.mean(skipna=True)),
                'std': float(s.std(skipna=True)) if s.count() > 1 else 0.0,
                'min': float(s.min(skipna=True)) if s.count() > 0 else float('nan'),
                'max': float(s.max(skipna=True)) if s.count() > 0 else float('nan'),
            })
        if not rows:
            LOGGER.warning("CSV de histórico não possui as métricas esperadas.")
            return None
        out_df = pd.DataFrame(rows)
        out_path = os.path.join(RESULTS_DIR, 'history', f"{nome_modelo}_history_summary.csv")
        out_df.to_csv(out_path, index=False)
        LOGGER.info(f"Resumo de histórico (via CSV) salvo em: {out_path}")
        return out_path
    except Exception as e:
        LOGGER.warning(f"Falha ao resumir histórico a partir do CSV: {e}")
        return None
