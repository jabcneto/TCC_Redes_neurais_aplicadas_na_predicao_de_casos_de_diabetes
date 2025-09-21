# Predição de Diabetes com Aprendizado de Máquina e Deep Learning

Projeto do TCC que implementa um pipeline completo e reprodutível para prever diabetes a partir de variáveis clínicas, combinando modelos clássicos de Machine Learning e arquiteturas de Deep Learning (Keras/TensorFlow). O pipeline cobre desde o carregamento dos dados, análise exploratória, pré-processamento e balanceamento de classes, até treinamento, avaliação, comparação de modelos e interpretabilidade (SHAP e LIME).

## Objetivos
- Construir e avaliar modelos preditivos de diabetes com métricas robustas (ROC AUC, F1, Precision/Recall, etc.).
- Comparar modelos clássicos e redes neurais em um cenário tabular.
- Explicar as previsões por meio de interpretabilidade (SHAP e LIME) para suporte à decisão clínica.

## Dados
- Arquivo: `diabetes_prediction_dataset.csv` (incluído no repositório).
- Alvo: `diabetes` (0 = não, 1 = sim).
- Variáveis (exemplos comuns no conjunto):
  - Categóricas: `gender`, `smoking_history`.
  - Numéricas: `age`, `bmi`, `HbA1c_level`, `blood_glucose_level`, além de indicadores binários como `hypertension`, `heart_disease`.

Observação: Ajuste a descrição acima conforme a versão final do seu dataset (fonte, licença e dicionário de dados). Se necessário, adicione a referência bibliográfica no final.

## Metodologia
1. Análise exploratória
   - Filtra `gender == 'Other'` (se existir).
   - Gera gráficos básicos de distribuição da variável alvo e salva em `resultados_diabetes/graficos/distribuicao`.
2. Pré-processamento
   - Split estratificado: treino/validação/teste (80/10/10).
   - Tratamento de outliers em colunas numéricas via IQR capping (clipping por limites Q1−1.5*IQR e Q3+1.5*IQR aprendidos no treino).
   - Codificação One-Hot (drop='first', handle_unknown='ignore').
   - Padronização com StandardScaler.
   - Balanceamento no conjunto de treino com SMOTE.
3. Modelagem
   - Modelos clássicos: Regressão Logística, Random Forest, Gradient Boosting, XGBoost.
   - Modelos Keras: MLP densa, CNN 1D para dados tabulares, e arquitetura híbrida CNN-LSTM.
4. Treinamento e seleção
   - Clássicos: ajuste direto e salvamento em `.pkl`.
   - Keras: callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard) salvando melhor modelo (`*_best.keras`) e final (`*_final.keras`).
5. Avaliação e comparação
   - Métricas: Accuracy, Balanced Accuracy, ROC AUC, PR AUC, F1, Precision, Recall, Specificity, FPR, FNR, NPV, MCC, G-Mean, Brier Score, Log Loss.
   - Gráficos: matriz de confusão, ROC, Precision-Recall, calibração.
   - Comparação consolidada por ROC AUC em `resultados_diabetes/comparacao_modelos_roc_auc.png`.
6. Interpretabilidade
   - SHAP: `resultados_diabetes/graficos/shap/*` (TreeExplainer para árvores, Deep/Kernel para redes; fallback automático).
   - LIME: `resultados_diabetes/graficos/lime/*.html` para algumas amostras do teste.

## Estrutura principal
- `main.py`: orquestra o pipeline, treina (opcional) e avalia.
- `config.py`: logging colorido, diretórios de saída, utilitários.
- `data_processing.py`: EDA, split, IQR capping, One-Hot, scaler, SMOTE.
- `modeling.py`: definição dos modelos clássicos e redes (MLP, CNN 1D, Híbrido CNN-LSTM).
- `training.py`: rotina de treino, callbacks e salvamento.
- `evaluation.py`: métricas, gráficos e comparação.
- `interpretability.py`: SHAP e LIME para explicabilidade.
- `utils.py`: constantes e seeds (reprodutibilidade).
- `resultados_diabetes/`: saídas do pipeline (modelos, gráficos, logs e históricos).

## Requisitos
- Python 3.10
- Dependências em `requirements.txt` (inclui scikit-learn, imbalanced-learn, xgboost, tensorflow, shap, lime, seaborn, etc.).

Instalação (ambiente isolado recomendado):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dica: Para usar GPUs com TensorFlow, siga a documentação oficial (versões de CUDA/cuDNN compatíveis).

## Como executar
- Executar apenas avaliação usando modelos já salvos (se existirem):
```bash
python main.py
```
- Forçar retreinamento de todos os modelos:
```bash
python main.py --retrain
```

Comportamento do `main.py`:
- Sem `--retrain`: carrega modelos clássicos de `resultados_diabetes/modelos/*.pkl` e redes de `*.keras` (se TensorFlow estiver disponível). Caso algum modelo clássico não exista, ele é automaticamente treinado.
- Com `--retrain`: treina todos os modelos clássicos; se TensorFlow estiver disponível, também treina MLP, CNN e Híbrido CNN-LSTM.

## Saídas geradas
- `resultados_diabetes/modelos/`
  - Clássicos: `regressão_logística.pkl`, `random_forest.pkl`, `gradient_boosting.pkl`, `xgboost.pkl`.
  - Keras: `MLP_best.keras`, `CNN_best.keras`, `Hibrido_CNN_LSTM_best.keras` (+ versões `_final.keras`).
- `resultados_diabetes/graficos/`
  - `confusao/cm_<modelo>.png`, `roc/roc_<modelo>.png`, `pr/pr_<modelo>.png`, `calibracao/cal_<modelo>.png`, `shap/*.png`, `lime/*.html`.
- `resultados_diabetes/history/`: CSVs de histórico de treino Keras.
- `resultados_diabetes/logs/`: logs globais e de sessão (TensorBoard para redes).
- CSVs de métricas por modelo na raiz de `resultados_diabetes`.
- `comparacao_modelos_roc_auc.png` com o ranking por ROC AUC.

## Métricas e gráficos
Cada modelo avaliado gera:
- CSV com métricas consolidadas.
- Matriz de confusão, curvas ROC e Precision-Recall, e curva de calibração.
- Comparativo final por ROC AUC. Use também F1, Precision, Recall e MCC para avaliar trade-offs.

## Interpretabilidade (SHAP e LIME)
- SHAP: identifica as variáveis que mais impactam a probabilidade prevista; o código seleciona automaticamente o melhor tipo de explicador conforme o modelo.
- LIME: explica localmente algumas amostras do conjunto de teste, salvando HTMLs com as contribuições de atributos.

## Reprodutibilidade
- Semente fixa: `RANDOM_STATE = 42`.
- Pipeline determinístico sempre que possível; variações podem ocorrer em algoritmos paralelos/aleatórios e no TensorFlow.
- Todos os artefatos são salvos sob `resultados_diabetes/` para posterior auditoria.

## Limitações e cuidados
- Dados tabulares podem não se beneficiar de CNN/LSTM; compare com modelos de árvores e regressão.
- Desequilíbrio de classes é tratado com SMOTE apenas no treino; avalie também métricas sensíveis a classe minoritária.
- Interpretações (SHAP/LIME) devem ser vistas como apoio, não como verdade causal.

## Problemas comuns (troubleshooting)
- TensorFlow indisponível: o pipeline segue com modelos clássicos e registra aviso. Para treinar/avaliar redes, instale TensorFlow compatível.
- Falta de modelos salvos: sem `--retrain`, os clássicos ausentes serão treinados automaticamente; redes precisam ser treinadas com `--retrain`.
- Erros de importação (xgboost, shap, lime): confirme `pip install -r requirements.txt` no ambiente ativo.

## Como citar no TCC (sugestão)
- Descreva o pipeline, as etapas de pré-processamento e os modelos conforme seções acima.
- Reporte as métricas por modelo e inclua os gráficos gerados nas pastas `graficos/*`.
- Inclua uma discussão sobre vieses, calibragem e interpretabilidade.

## Licença e créditos
- Licença do código: defina no arquivo `LICENSE` conforme a sua necessidade (p.ex., MIT).
- Dataset: descreva a fonte e a licença do conjunto de dados utilizado, se aplicável.
