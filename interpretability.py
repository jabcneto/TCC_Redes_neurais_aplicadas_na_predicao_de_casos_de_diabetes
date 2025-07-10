# interpretability.py
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import LOGGER
from utils import RESULTS_DIR


def analisar_valores_shap(model, x_test, feature_names, nome_modelo):
    """Calcula e plota os valores SHAP para um modelo."""
    LOGGER.info(f"Calculando valores SHAP para {nome_modelo}.")
    save_dir = os.path.join(RESULTS_DIR, "graficos", "shap")

    # SHAP precisa de um subconjunto de dados para o explainer
    explainer = shap.KernelExplainer(model.predict, shap.sample(x_test, 100))
    shap_values = explainer.shap_values(x_test)

    # Gráfico de resumo
    shap.summary_plot(shap_values, x_test, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(save_dir, f"shap_summary_{nome_modelo}.png"), dpi=300)
    plt.close()


def analisar_lime_pt(model, x_train, x_test, feature_names, nome_modelo, num_amostras=3):
    """Gera explicações LIME para algumas amostras."""
    LOGGER.info(f"Gerando LIME para {nome_modelo}.")
    save_dir = os.path.join(RESULTS_DIR, "graficos", "lime")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        x_train,
        feature_names=feature_names,
        class_names=['Não Diabetes', 'Diabetes'],
        mode='classification'
    )

    # Função de predição para o LIME
    def predict_fn(x):
        return model.predict(x)

    # Gerar explicações para algumas amostras aleatórias
    for i in np.random.choice(range(len(x_test)), num_amostras, replace=False):
        exp = explainer.explain_instance(x_test[i], predict_fn, num_features=10)
        exp.save_to_file(os.path.join(save_dir, f"lime_{nome_modelo}_amostra_{i}.html"))