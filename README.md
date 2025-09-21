# Predição de Diabetes com Aprendizado de Máquina e Deep Learning

Projeto do TCC que implementa um pipeline completo e reprodutível para prever diabetes a partir de variáveis clínicas, combinando modelos clássicos de Machine Learning e arquiteturas de Deep Learning (Keras/TensorFlow). O pipeline cobre desde o carregamento dos dados, análise exploratória, pré-processamento e balanceamento de classes, até treinamento, avaliação, comparação de modelos e interpretabilidade (SHAP e LIME).

## Objetivos
- Construir e avaliar modelos preditivos de diabetes com métricas robustas (ROC AUC, F1, Precision/Recall, etc.).
- Comparar modelos clássicos e redes neurais em um cenário tabular.
- Explicar as previsões por meio de interpretabilidade (SHAP e LIME) para suporte à decisão clínica.

## Dados
- Arquivo: `diabetes_prediction_dataset.csv`.
- Alvo: `diabetes` (0 = não, 1 = sim).
- Exemplos de variáveis: `gender`, `smoking_history`, `age`, `bmi`, `HbA1c_level`, `blood_glucose_level`, `hypertension`, `heart_disease`.

## Metodologia
1. Análise exploratória
   - Geração de gráficos de distribuição e correlação em `resultados_diabetes/graficos/distribuicao`.
2. Pré-processamento
   - Split estratificado: treino/validação/teste.
   - Tratamento de outliers via IQR capping.
   - One-Hot Encoding e padronização com StandardScaler.
   - Balanceamento no treino com SMOTE.
3. Modelagem
   - Clássicos: Regressão Logística, Random Forest, Gradient Boosting, XGBoost.
   - Keras: MLP, CNN 1D, Híbrido CNN-LSTM.
4. Treinamento e seleção
   - Clássicos: ajuste e salvamento `.pkl`.
   - Keras: callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger), salvando `*_best.keras` e `*_final.keras`.
5. Avaliação e comparação
   - Métricas principais salvas por modelo (CSV) e gráficos individuais.
   - Comparação consolidada por ROC AUC em `resultados_diabetes/comparacao_modelos_roc_auc.png`.
6. Interpretabilidade
   - SHAP e LIME para explicações globais e locais.

## Estrutura principal
- `main.py`: orquestra o pipeline.
- `config.py`: logging e diretórios de saída.
- `data_processing.py`: EDA, split, capping, One-Hot, scaler, SMOTE.
- `modeling.py`: definição dos modelos.
- `training.py`: rotinas de treino.
- `evaluation.py`: métricas e gráficos de avaliação.
- `interpretability.py`: SHAP e LIME.
- `resultados_diabetes/`: saídas do pipeline.

## Saídas geradas
- `resultados_diabetes/modelos/`: artefatos `.pkl` e `.keras`.
- `resultados_diabetes/graficos/`:
  - `confusao/`: `matriz_confusao_<modelo>.png` e normalizada.
  - `roc/`: `roc_<modelo>.png`.
  - `pr/`: `pr_<modelo>.png`.
  - `calibracao/`: `cal_<modelo>.png`.
  - `metricas/`: `acuracia_<modelo>.png`, `precisao_<modelo>.png`, `recall_<modelo>.png`, `f1_score_<modelo>.png`, `mcc_<modelo>.png`, `auc_roc_<modelo>.png`, `brier_score_<modelo>.png`.
  - `importancia/`, `distribuicao/`, `shap/`, `lime/`.
- CSVs de métricas por modelo na raiz de `resultados_diabetes`.

## Métricas
- Discriminação: AUC ROC, PR AUC, KS, Gini.
- Classificação: Acurácia, Balanced Accuracy, Precisão, Recall, F1 (F1, F2, F0.5), MCC, G-Mean.
- Erro/Probabilidade: Brier Score, Log Loss, ECE/MCE, interceto e inclinação de calibração.
- Confusão: Especificidade, FPR, FNR, NPV, FDR, FOR, Youden J, LR+/LR-.

## Execução
```bat
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py --retrain
```

## Notas
- Nomes e rótulos dos gráficos padronizados em português.
- As pastas necessárias são criadas automaticamente no início do pipeline.
