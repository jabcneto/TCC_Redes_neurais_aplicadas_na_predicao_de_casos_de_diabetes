import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import os
from config import LOGGER, RESULTS_DIR


def analisar_valores_shap(model, x_test, feature_names, nome_modelo):
    LOGGER.info(f"Calculando valores SHAP para {nome_modelo}.")
    save_dir = os.path.join(RESULTS_DIR, "graficos", "shap")

    def _keras_predict_fn(X):
        if len(model.input_shape) == 3 and X.ndim == 2:
            X = np.expand_dims(X, axis=-1)
        preds = model.predict(X)
        return preds.flatten()

    is_keras = hasattr(model, "input_shape")

    try:
        import shap
        X = x_test
        if isinstance(X, np.ndarray) and X.ndim == 3 and X.shape[-1] == 1:
            X = X.squeeze(-1)
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        k = min(50, max(10, n // 100))

        if is_keras:
            background = shap.kmeans(X, k)
            try:
                if len(model.input_shape) == 2:
                    explainer = shap.DeepExplainer(model, background)
                    X_sample = shap.sample(X, min(300, n))
                    shap_values = explainer.shap_values(X_sample)
                else:
                    explainer = shap.KernelExplainer(_keras_predict_fn, background)
                    X_sample = shap.sample(X, min(200, n))
                    shap_values = explainer.shap_values(X_sample)
            except Exception:
                explainer = shap.KernelExplainer(_keras_predict_fn, background)
                X_sample = shap.sample(X, min(200, n))
                shap_values = explainer.shap_values(X_sample)
        else:
            cls = model.__class__.__name__.lower()
            try:
                if any(name in cls for name in ["xgb", "randomforest", "gradientboosting", "xgboost"]):
                    explainer = shap.TreeExplainer(model)
                    X_sample = shap.sample(X, min(2000, n))
                    shap_values = explainer.shap_values(X_sample)
                elif "logisticregression" in cls:
                    background = shap.kmeans(X, k)
                    explainer = shap.LinearExplainer(model, background)
                    X_sample = shap.sample(X, min(2000, n))
                    shap_values = explainer.shap_values(X_sample)
                else:
                    background = shap.kmeans(X, k)
                    def _sk_predict_fn(A):
                        return model.predict_proba(A)[:, 1]
                    explainer = shap.KernelExplainer(_sk_predict_fn, background)
                    X_sample = shap.sample(X, min(500, n))
                    shap_values = explainer.shap_values(X_sample)
            except Exception:
                background = shap.kmeans(X, k)
                def _sk_predict_fn(A):
                    return model.predict_proba(A)[:, 1]
                explainer = shap.KernelExplainer(_sk_predict_fn, background)
                X_sample = shap.sample(X, min(500, n))
                shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            try:
                shap_values = shap_values[1]
            except Exception:
                shap_values = shap_values[0]

        max_disp = min(10, len(feature_names))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=max_disp)
        plt.savefig(os.path.join(save_dir, f"shap_summary_{nome_modelo}.png"), dpi=300)
        plt.close()
    except Exception as e:
        LOGGER.error(f"Falha ao calcular SHAP para {nome_modelo}: {e}")


def analisar_lime_pt(model, x_train, x_test, feature_names, nome_modelo, num_amostras=3):
    LOGGER.info(f"Gerando LIME para {nome_modelo}.")
    save_dir = os.path.join(RESULTS_DIR, "graficos", "lime")

    if isinstance(x_train, np.ndarray) and x_train.ndim == 3 and x_train.shape[-1] == 1:
        x_train_2d = x_train.squeeze(-1)
    else:
        x_train_2d = x_train

    explainer = lime.lime_tabular.LimeTabularExplainer(
        x_train_2d,
        feature_names=feature_names,
        class_names=['Não Diabetes', 'Diabetes'],
        mode='classification'
    )

    def predict_fn(X):
        if hasattr(model, "input_shape"):
            if X.ndim == 2 and len(model.input_shape) == 3:
                X = np.expand_dims(X, axis=-1)
            probs = model.predict(X).flatten()
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return np.column_stack([1 - probs, probs])
        else:
            p = model.predict_proba(X)[:, 1]
            return np.column_stack([1 - p, p])

    n = len(x_test)
    for i in np.random.choice(range(n), min(num_amostras, n), replace=False):
        xi = x_test[i]
        if isinstance(xi, np.ndarray) and xi.ndim == 2 and xi.shape[-1] == 1:
            xi = xi.squeeze(-1)
        exp = explainer.explain_instance(xi, predict_fn, num_features=min(10, len(feature_names)))
        exp.save_to_file(os.path.join(save_dir, f"lime_{nome_modelo}_amostra_{i}.html"))