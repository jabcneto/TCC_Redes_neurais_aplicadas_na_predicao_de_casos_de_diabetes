# Arquitetura CNN para Predição de Diabetes

## Visão Geral

A infraestrutura CNN foi implementada seguindo o mesmo padrão da MLP, com arquivos espelhados para manter consistência no código.

## Arquivos Criados

### 1. `cnn_utils.py`
Funções auxiliares para CNN:
- `create_cnn_from_hyperparameters()` - Cria CNN a partir de dicionário de hiperparâmetros
- `retrain_final_cnn()` - Retreina CNN final com todos os dados
- `load_best_cnn_model()` - Carrega melhor modelo CNN disponível

### 2. `cnn_tuning.py`
Otimização de hiperparâmetros:
- `build_tunable_cnn()` - Define espaço de busca de hiperparâmetros
- `tune_cnn_hyperparameters()` - Busca RandomSearch
- `bayesian_tune_cnn()` - Busca Bayesiana
- `create_cnn_from_best_hps()` - Cria modelo a partir dos melhores HPs
- `analyze_cnn_tuning_results()` - Analisa resultados do tuning

## Arquitetura da CNN

A CNN 1D implementada possui:

### Camadas Convolucionais (1-3 camadas)
- **Filtros**: 16 a 128 (incrementos de 16)
- **Kernel Size**: 3, 5 ou 7
- **Ativação**: relu, elu ou selu
- **Regularização L2**: 1e-5 a 1e-2 (log scale)
- **Batch Normalization**: opcional
- **MaxPooling**: tamanho 2 ou 3 (opcional)
- **Dropout**: 0.1 a 0.5

### Camadas Densas (1-3 camadas)
- **Unidades**: 32 a 256 (incrementos de 32)
- **Ativação**: relu, elu ou selu
- **Regularização L2**: 1e-5 a 1e-2
- **Batch Normalization**: opcional
- **Dropout**: 0.1 a 0.5

### Camada de Saída
- 1 neurônio com ativação sigmoid (classificação binária)

### Otimizadores
- Adam, Nadam, RMSprop ou SGD
- Learning rate: 1e-5 a 1e-2 (log scale)

## Uso

### Buscar Hiperparâmetros CNN (RandomSearch)
```bash
python main.py --tune-cnn --trials 50
```

### Buscar Hiperparâmetros CNN (Bayesian)
```bash
python main.py --bayesian-cnn --trials 30
```

### Validação Cruzada com CNN
```bash
python main.py --cv
```

### Avaliar Todos os Modelos (incluindo CNN)
```bash
python main.py
```

## Integração com Pipeline Existente

A CNN foi integrada em todos os componentes:

1. **model_management.py** - `evaluate_cnn_models()`
2. **comparison_utils.py** - `_add_cnn_metrics()`
3. **tuning_pipelines.py** - `run_cnn_hyperparameter_tuning()`, `run_cnn_bayesian_tuning()`
4. **cv_pipelines.py** - `run_cnn_cross_validation_with_pretrained()`
5. **main.py** - `run_cnn_tuning_pipeline()`

## Modelos Salvos

Os modelos CNN são salvos em:
- `resultados_diabetes/modelos/CNN_Tuned_Final.keras`
- `resultados_diabetes/modelos/CNN_Bayesian_Final.keras`

## Configurações de Tuning

As configurações são salvas em:
- `resultados_diabetes/tuning/cnn_results/best_trial_config.json`
- `resultados_diabetes/tuning/cnn_bayesian_results/bayesian_best_config.json`

## Diferenças entre MLP e CNN

### MLP
- Camadas densas totalmente conectadas
- Processa features de forma independente
- Mais simples, menos parâmetros

### CNN 1D
- Camadas convolucionais que aprendem padrões locais
- Detecta correlações entre features adjacentes
- MaxPooling reduz dimensionalidade
- Mais parâmetros, potencialmente mais expressiva

## Exemplo de Fluxo Completo

```bash
# 1. Otimizar hiperparâmetros da CNN
python main.py --bayesian-cnn --trials 30

# 2. Avaliar todos os modelos (incluindo CNN)
python main.py

# 3. Validação cruzada
python main.py --cv
```

## Métricas de Avaliação

Todas as métricas padrão são calculadas:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Balanced Accuracy

## Estrutura de Pastas

```
resultados_diabetes/
├── modelos/
│   ├── CNN_Tuned_Final.keras
│   └── CNN_Bayesian_Final.keras
├── tuning/
│   ├── cnn_results/
│   │   ├── best_trial_config.json
│   │   └── cnn_tuning/
│   └── cnn_bayesian_results/
│       ├── bayesian_best_config.json
│       └── cnn_bayesian_tuning/
└── graficos/
    └── (gráficos comparativos incluem CNN)
```

