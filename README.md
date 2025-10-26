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
   - Balanceamento no treino com SMOTE (quando necessário).
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
- `main.py`: orquestra o pipeline e a CLI simplificada.
- `config.py`: logging, parâmetros padrão e diretórios de saída.
- `data_processing.py`: EDA, split, capping, One-Hot, scaler, reamostragem.
- `modeling.py`: definição dos modelos.
- `training.py`: rotinas de treino.
- `evaluation.py`: métricas e gráficos de avaliação.
- `tuning_pipelines.py`, `bayesian_tuning.py`, `cnn_tuning.py`: buscas bayesianas e utilitários.
- `resultados_diabetes/`: saídas do pipeline.

## Saídas geradas
- `resultados_diabetes/modelos/`: artefatos `.pkl` e `.keras`.
- `resultados_diabetes/graficos/`:
  - `confusao/`: `cm_<modelo>.png`.
  - `roc/`: `roc_<modelo>.png`.
  - `pr/`: `pr_<modelo>.png`.
  - `distribuicao/`: `dist_target.png` e outros.
- `resultados_diabetes/tuning/`:
  - `bayesian_results/` (MLP) e `cnn_bayesian_results/` (CNN) com melhores configs e históricos.
  - Consolidados: `consolidated_trials_summary.csv` e `consolidated_epoch_history.csv`.
- `resultados_diabetes/logs/`: logs de execução com timestamp.

## Parâmetros padrão (config.py)
- `DEFAULT_EPOCHS = 100`
- `DEFAULT_TUNING_EPOCHS = 25`
- `DEFAULT_FINAL_TRAINING_EPOCHS = 80`
- `DEFAULT_BATCH_SIZE = 128`

Ajuste esses valores em `config.py` conforme necessidade. Callbacks de EarlyStopping encurtam o treino quando não há melhoria.

## Execução (Windows - cmd.exe)

Criar ambiente e instalar dependências:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Avaliar modelos já treinados:

```bat
python main.py
```

Tuning bayesiano de ambos (CNN + MLP):

```bat
python main.py --bayesian --trials 30
```

Tuning apenas MLP:

```bat
python main.py --bayesian-mlp --trials 15
```

Tuning apenas CNN:

```bat
python main.py --bayesian-cnn --trials 15
```

Retreinar MLP a partir de um trial (por número):

```bat
python main.py --train-mlp-trial-number 7
```

Retreinar MLP a partir de um trial (por id):

```bat
python main.py --train-mlp-trial-id 0007
```

Retreinar CNN a partir de um trial.json do Keras Tuner:

```bat
python main.py --train-cnn-trial-json resultados_diabetes\tuning\cnn_bayesian_results\cnn_bayesian_tuning\trial_00\trial.json
```

## Tuning separado (CNN vs MLP)
- Use `--bayesian-mlp` para otimizar somente o MLP.
- Use `--bayesian-cnn` para otimizar somente a CNN.
- `--bayesian` mantém o comportamento de otimizar ambos, em sequência.

## Dicas de desempenho e estabilidade (WSL/Windows)
- Comece com valores conservadores e aumente gradualmente:
  - Trials: 10–15 por modelo; se estável, aumentar para 20–30.
  - `DEFAULT_TUNING_EPOCHS`: 15–25 costuma ser suficiente para o tuner.
  - `DEFAULT_FINAL_TRAINING_EPOCHS`: 50–100 para o retreino final (EarlyStopping ativo).
- Em ambientes WSL, “Killed” geralmente indica falta de memória. O código libera sessão do TF entre trials, reduzindo o risco. Ainda assim:
  - Reduza `--trials` e `DEFAULT_TUNING_EPOCHS` em testes.
  - Evite rodar CNN e MLP juntos se a memória for limitada; use as flags separadas.
  - Diminua `DEFAULT_BATCH_SIZE` se necessário.
- Acompanhe logs em `resultados_diabetes/logs/` para estimativas de duração e progresso dos trials.

## Troubleshooting
- Processo encerrado ("Killed"):
  - Rode CNN e MLP separadamente; reduza trials/epochs/batch size.
  - Feche aplicações que consumam muita RAM durante o tuning.
- Falha ao carregar hiperparâmetros para retreino:
  - Verifique os arquivos sob `resultados_diabetes/tuning/` e os parâmetros `--train-mlp-trial-*` ou `--train-cnn-trial-json`.
