import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import os
from config import LOGGER, RESULTS_DIR


def analisar_valores_shap(model, x_test, feature_names, nome_modelo):
    LOGGER.info(f"Calculando valores SHAP para {nome_modelo}.")
    save_dir = os.path.join(RESULTS_DIR, "graficos", "shap")

    def predict_fn_shap(X):
        if len(model.input_shape) == 3 and X.ndim == 2:
            X = np.expand_dims(X, axis=-1)
        preds = model.predict(X)
        return preds.flatten()

    background = x_test
    if isinstance(background, np.ndarray) and background.ndim == 3 and background.shape[-1] == 1:
        background = background.squeeze(-1)
    if isinstance(x_test, np.ndarray) and x_test.ndim == 3 and x_test.shape[-1] == 1:
        x_for_shap = x_test.squeeze(-1)
    else:
        x_for_shap = x_test

    try:
        import shap
        explainer = shap.KernelExplainer(predict_fn_shap, shap.sample(x_for_shap, 100))
        shap_values = explainer.shap_values(x_for_shap)
        shap.summary_plot(shap_values, x_for_shap, feature_names=feature_names, show=False)
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
        if len(model.input_shape) == 3 and X.ndim == 2:
            X = np.expand_dims(X, axis=-1)
        probs = model.predict(X).flatten()
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return np.column_stack([1 - probs, probs])

    n = len(x_test)
    for i in np.random.choice(range(n), min(num_amostras, n), replace=False):
        xi = x_test[i]
        if isinstance(xi, np.ndarray) and xi.ndim == 2 and xi.shape[-1] == 1:
            xi = xi.squeeze(-1)
        exp = explainer.explain_instance(xi, predict_fn, num_features=min(10, len(feature_names)))
        exp.save_to_file(os.path.join(save_dir, f"lime_{nome_modelo}_amostra_{i}.html"))