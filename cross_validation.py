import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from config import RESULTS_DIR, LOGGER, DEFAULT_TUNING_EPOCHS, DEFAULT_BATCH_SIZE
from utils import compute_class_weights_from_labels, find_optimal_threshold


def cross_validate_mlp(model_builder, x_train, y_train, n_folds=5, epochs=100, batch_size=64, patience=15):
    LOGGER.info("=" * 80)
    LOGGER.info(f"INICIANDO VALIDAÇÃO CRUZADA COM {n_folds} FOLDS")
    LOGGER.info("=" * 80)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    fold_histories = []
    fold_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train), 1):
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info(f"FOLD {fold_idx}/{n_folds}")
        LOGGER.info("=" * 80)

        x_fold_train, x_fold_val = x_train[train_idx], x_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        LOGGER.info(f"Train size: {len(x_fold_train)}")
        LOGGER.info(f"Validation size: {len(x_fold_val)}")
        LOGGER.info(f"Train positive ratio: {y_fold_train.sum() / len(y_fold_train):.2%}")
        LOGGER.info(f"Val positive ratio: {y_fold_val.sum() / len(y_fold_val):.2%}")

        model = model_builder()

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        class FoldProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                if (epoch + 1) % 10 == 0:
                    LOGGER.info(
                        f"  Época {epoch+1}/{epochs}: "
                        f"loss={logs.get('loss', 0):.4f}, "
                        f"val_loss={logs.get('val_loss', 0):.4f}, "
                        f"val_precision={logs.get('val_precision', 0):.4f}, "
                        f"val_accuracy={logs.get('val_accuracy', 0):.4f}, "
                        f"val_pr_auc={logs.get('val_pr_auc', 0):.4f}"
                    )

        fold_progress = FoldProgressCallback()

        LOGGER.info(f"Iniciando treinamento do Fold {fold_idx}...")
        pos_ratio = y_fold_train.mean()
        use_class_weight = not (0.45 <= pos_ratio <= 0.55)
        class_weight = compute_class_weights_from_labels(y_fold_train) if use_class_weight else None
        if not use_class_weight:
            LOGGER.info("Fold praticamente balanceado; class_weight desativado para evitar sobreajuste de threshold.")
        history = model.fit(
            x_fold_train, y_fold_train,
            validation_data=(x_fold_val, y_fold_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, fold_progress],
            verbose=0,
            class_weight=class_weight
        )

        y_pred_proba = model.predict(x_fold_val, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, balanced_accuracy_score, average_precision_score
        )

        thr, thr_stats = find_optimal_threshold(y_fold_val, y_pred_proba, objective='f1')
        y_pred_opt = (y_pred_proba > thr).astype(int)

        fold_metrics = {
            'fold': fold_idx,
            'accuracy': accuracy_score(y_fold_val, y_pred),
            'precision': precision_score(y_fold_val, y_pred, zero_division=0),
            'recall': recall_score(y_fold_val, y_pred, zero_division=0),
            'f1_score': f1_score(y_fold_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_fold_val, y_pred_proba),
            'pr_auc': average_precision_score(y_fold_val, y_pred_proba),
            'balanced_accuracy': balanced_accuracy_score(y_fold_val, y_pred),
            'opt_threshold': thr,
            'precision_opt': precision_score(y_fold_val, y_pred_opt, zero_division=0),
            'recall_opt': recall_score(y_fold_val, y_pred_opt, zero_division=0),
            'f1_opt': f1_score(y_fold_val, y_pred_opt, zero_division=0),
            'train_samples': len(x_fold_train),
            'val_samples': len(x_fold_val),
            'epochs_trained': len(history.history['loss'])
        }

        fold_results.append(fold_metrics)
        fold_histories.append(history.history)
        fold_models.append(model)

        LOGGER.info(f"\nFold {fold_idx} concluído!")
        LOGGER.info(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
        LOGGER.info(f"  Precision (thr=0.5): {fold_metrics['precision']:.4f}")
        LOGGER.info(f"  Recall (thr=0.5): {fold_metrics['recall']:.4f}")
        LOGGER.info(f"  F1-Score (thr=0.5): {fold_metrics['f1_score']:.4f}")
        LOGGER.info(f"  PR-AUC: {fold_metrics['pr_auc']:.4f}")
        LOGGER.info(f"  Threshold ótimo (F1): {thr:.3f} -> Precision={fold_metrics['precision_opt']:.4f}, Recall={fold_metrics['recall_opt']:.4f}, F1={fold_metrics['f1_opt']:.4f}")

    df_results = pd.DataFrame(fold_results)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("RESULTADOS DA VALIDAÇÃO CRUZADA")
    LOGGER.info("=" * 80)

    metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'balanced_accuracy', 'precision_opt', 'recall_opt', 'f1_opt']

    LOGGER.info("\n📊 Média e Desvio Padrão por Métrica:")
    for metric in metrics_to_show:
        mean_val = df_results[metric].mean()
        std_val = df_results[metric].std()
        LOGGER.info(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

    cv_results_dir = os.path.join(RESULTS_DIR, 'cross_validation')
    os.makedirs(cv_results_dir, exist_ok=True)

    csv_path = os.path.join(cv_results_dir, 'cv_results.csv')
    df_results.to_csv(csv_path, index=False)
    LOGGER.info(f"\n✓ Resultados salvos em: {csv_path}")

    summary = {
        'n_folds': n_folds,
        'mean_metrics': df_results[metrics_to_show].mean().to_dict(),
        'std_metrics': df_results[metrics_to_show].std().to_dict(),
        'fold_results': fold_results
    }

    json_path = os.path.join(cv_results_dir, 'cv_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    LOGGER.info(f"✓ Resumo salvo em: {json_path}")

    best_fold_idx = df_results['f1_opt'].idxmax()
    best_model = fold_models[best_fold_idx]

    LOGGER.info(f"\n🏆 Melhor Fold: {best_fold_idx + 1}")
    LOGGER.info(f"  F1-OPT: {df_results.loc[best_fold_idx, 'f1_opt']:.4f}")
    LOGGER.info(f"  Precision-OPT: {df_results.loc[best_fold_idx, 'precision_opt']:.4f}")
    LOGGER.info(f"  Recall-OPT: {df_results.loc[best_fold_idx, 'recall_opt']:.4f}")

    best_model_path = os.path.join(cv_results_dir, 'best_fold_model.keras')
    best_model.save(best_model_path)
    LOGGER.info(f"\n✓ Melhor modelo (Fold {best_fold_idx + 1}) salvo em: {best_model_path}")

    return {
        'fold_results': df_results,
        'fold_histories': fold_histories,
        'fold_models': fold_models,
        'best_model': best_model,
        'best_fold_idx': best_fold_idx,
        'summary': summary
    }


def cross_validate_with_tuning(model_builder_func, best_hps, x_train, y_train, n_folds=5, epochs=100, batch_size=64):
    LOGGER.info("=" * 80)
    LOGGER.info(f"VALIDAÇÃO CRUZADA COM HIPERPARÂMETROS OTIMIZADOS")
    LOGGER.info("=" * 80)

    def create_model():
        input_shape = (x_train.shape[1],)

        if hasattr(best_hps, 'values'):
            hps_values = best_hps.values
        else:
            hps_values = best_hps

        if 'num_conv_layers' in hps_values:
            from cnn_tuning import create_cnn_from_best_hps
            LOGGER.info("Detectado modelo CNN")
            return create_cnn_from_best_hps(best_hps, input_shape)
        else:
            from hyperparameter_tuning import create_model_from_best_hps
            LOGGER.info("Detectado modelo MLP")
            return create_model_from_best_hps(best_hps, input_shape)

    return cross_validate_mlp(
        model_builder=create_model,
        x_train=x_train,
        y_train=y_train,
        n_folds=n_folds,
        epochs=epochs,
        batch_size=batch_size
    )


def compare_cv_with_holdout(cv_results, holdout_metrics, model_name="modelo"):
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"COMPARAÇÃO: VALIDAÇÃO CRUZADA vs HOLDOUT - {model_name}")
    LOGGER.info("=" * 80)

    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    comparison = []
    for metric in metrics_to_compare:
        cv_mean = cv_results['fold_results'][metric].mean()
        cv_std = cv_results['fold_results'][metric].std()
        holdout_val = holdout_metrics.get(metric, 0)

        comparison.append({
            'metric': metric,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'holdout': holdout_val,
            'difference': abs(cv_mean - holdout_val),
            'model': model_name
        })

        LOGGER.info(f"\n{metric.upper()}:")
        LOGGER.info(f"  CV: {cv_mean:.4f} ± {cv_std:.4f}")
        LOGGER.info(f"  Holdout: {holdout_val:.4f}")
        LOGGER.info(f"  Diferença: {abs(cv_mean - holdout_val):.4f}")

    df_comparison = pd.DataFrame(comparison)

    cv_results_dir = os.path.join(RESULTS_DIR, 'cross_validation')
    os.makedirs(cv_results_dir, exist_ok=True)

    comparison_path = os.path.join(cv_results_dir, f'cv_vs_holdout_{model_name}.csv')
    df_comparison.to_csv(comparison_path, index=False)
    LOGGER.info(f"\n✓ Comparação salva em: {comparison_path}")

    return df_comparison


def nested_cross_validation(model_builder_func, hyperparameter_space, x_train, y_train,
                            outer_folds=5, inner_folds=3, max_trials=20,
                            epochs=None, batch_size=None, verbose=1):
    LOGGER.info("=" * 80)
    LOGGER.info(f"VALIDAÇÃO CRUZADA ANINHADA (NESTED CV)")
    LOGGER.info(f"Outer folds: {outer_folds}, Inner folds: {inner_folds}")
    LOGGER.info("=" * 80)

    # validação e defaults
    if not isinstance(outer_folds, int) or outer_folds <= 1:
        outer_folds = 5
    if not isinstance(inner_folds, int) or inner_folds <= 1:
        inner_folds = 3
    if epochs is None or not isinstance(epochs, int) or epochs <= 0:
        epochs = DEFAULT_TUNING_EPOCHS
    if batch_size is None or not isinstance(batch_size, int) or batch_size <= 0:
        batch_size = DEFAULT_BATCH_SIZE

    LOGGER.info(f"Parâmetros de tuning: epochs={epochs}, batch_size={batch_size}, verbose={verbose}")

    from sklearn.model_selection import StratifiedKFold
    import keras_tuner as kt

    outer_skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    outer_results = []

    for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_skf.split(x_train, y_train), 1):
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info(f"OUTER FOLD {outer_fold_idx}/{outer_folds}")
        LOGGER.info("=" * 80)

        x_outer_train, x_outer_test = x_train[train_idx], x_train[test_idx]
        y_outer_train, y_outer_test = y_train[train_idx], y_train[test_idx]

        LOGGER.info(f"Realizando busca de hiperparâmetros no outer fold {outer_fold_idx}...")

        tuner = kt.BayesianOptimization(
            model_builder_func,
            objective=kt.Objective('val_pr_auc', direction='max'),
            max_trials=max_trials,
            executions_per_trial=1,
            directory=os.path.join(RESULTS_DIR, 'nested_cv', f'outer_fold_{outer_fold_idx}'),
            project_name='inner_tuning',
            overwrite=True
        )

        inner_skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)
        inner_train_idx, inner_val_idx = next(inner_skf.split(x_outer_train, y_outer_train))

        x_inner_train = x_outer_train[inner_train_idx]
        y_inner_train = y_outer_train[inner_train_idx]
        x_inner_val = x_outer_train[inner_val_idx]
        y_inner_val = y_outer_train[inner_val_idx]

        early_stop = EarlyStopping(monitor='val_pr_auc', patience=10, mode='max', restore_best_weights=True)

        pos_ratio_inner = y_inner_train.mean()
        use_cw_inner = not (0.45 <= pos_ratio_inner <= 0.55)
        inner_class_weight = compute_class_weights_from_labels(y_inner_train) if use_cw_inner else None

        tuner.search(
            x_inner_train, y_inner_train,
            validation_data=(x_inner_val, y_inner_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=verbose,
            class_weight=inner_class_weight
        )

        best_model = tuner.get_best_models(num_models=1)[0]

        y_pred_proba = best_model.predict(x_outer_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, balanced_accuracy_score, average_precision_score
        )

        outer_fold_metrics = {
            'outer_fold': outer_fold_idx,
            'accuracy': accuracy_score(y_outer_test, y_pred),
            'precision': precision_score(y_outer_test, y_pred, zero_division=0),
            'recall': recall_score(y_outer_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_outer_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_outer_test, y_pred_proba),
            'pr_auc': average_precision_score(y_outer_test, y_pred_proba),
            'balanced_accuracy': balanced_accuracy_score(y_outer_test, y_pred)
        }

        outer_results.append(outer_fold_metrics)

        LOGGER.info(f"\nOuter Fold {outer_fold_idx} concluído!")
        LOGGER.info(f"  Accuracy: {outer_fold_metrics['accuracy']:.4f}")
        LOGGER.info(f"  Precision: {outer_fold_metrics['precision']:.4f}")
        LOGGER.info(f"  Recall: {outer_fold_metrics['recall']:.4f}")
        LOGGER.info(f"  F1-Score: {outer_fold_metrics['f1_score']:.4f}")
        LOGGER.info(f"  PR-AUC: {outer_fold_metrics['pr_auc']:.4f}")

    df_outer_results = pd.DataFrame(outer_results)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("RESULTADOS DA VALIDAÇÃO CRUZADA ANINHADA")
    LOGGER.info("=" * 80)

    metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'balanced_accuracy']

    LOGGER.info("\n📊 Estimativa de Generalização (Média ± DP):")
    for metric in metrics_to_show:
        mean_val = df_outer_results[metric].mean()
        std_val = df_outer_results[metric].std()
        LOGGER.info(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

    nested_cv_dir = os.path.join(RESULTS_DIR, 'nested_cv')
    os.makedirs(nested_cv_dir, exist_ok=True)

    results_path = os.path.join(nested_cv_dir, 'nested_cv_results.csv')
    df_outer_results.to_csv(results_path, index=False)
    LOGGER.info(f"\n✓ Resultados salvos em: {results_path}")

    return {
        'outer_results': df_outer_results,
        'mean_metrics': df_outer_results[metrics_to_show].mean().to_dict(),
        'std_metrics': df_outer_results[metrics_to_show].std().to_dict()
    }
