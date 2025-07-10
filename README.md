# Projeto de Predição de Diabetes

Este projeto implementa um pipeline completo de Machine Learning para predição de diabetes, utilizando uma variedade de modelos clássicos e de Deep Learning. O objetivo é demonstrar o processo de construção, treinamento, avaliação e interpretabilidade de modelos preditivos em um contexto de saúde.

## Estrutura do Projeto

O projeto está organizado nos seguintes módulos:

- `main.py`: Orquestra o pipeline completo, desde o carregamento dos dados até a interpretabilidade dos modelos.
- `config.py`: Contém configurações globais, como o logger, estado aleatório e caminhos de diretórios de resultados.
- `data_processing.py`: Responsável pelo carregamento, análise exploratória, pré-processamento (tratamento de outliers, escalonamento, codificação One-Hot, balanceamento com SMOTE) e divisão dos dados.
- `modeling.py`: Define e constrói os modelos de Machine Learning, incluindo Regressão Logística, Random Forest, Gradient Boosting, XGBoost, MLP, CNN 1D e um modelo Híbrido CNN-LSTM.
- `training.py`: Contém as funções para treinar os modelos clássicos e os modelos Keras, com callbacks para otimização do treinamento (Early Stopping, Model Checkpoint, ReduceLROnPlateau).
- `evaluation.py`: Implementa a avaliação dos modelos, gerando métricas de desempenho (accuracy, ROC AUC, F1-Score, Precision, Recall), matrizes de confusão e curvas ROC. Também inclui funções para comparação de modelos.
- `interpretability.py`: Foca na interpretabilidade dos modelos, utilizando técnicas como SHAP (SHapley Additive exPlanations) e LIME (Local Interpretable Model-agnostic Explanations) para entender como as previsões são feitas.
- `utils.py`: Contém funções utilitárias, como a criação de diretórios de projeto e a configuração do sistema de logging.
- `requirements.txt`: Lista todas as dependências Python necessárias para o projeto.
- `diabetes_prediction_dataset.csv`: O dataset utilizado para o treinamento e avaliação dos modelos.

## Modelos Implementados

O projeto explora os seguintes tipos de modelos:

### Modelos Clássicos
- Regressão Logística
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

### Modelos de Deep Learning (Keras/TensorFlow)
- **MLP (Multi-Layer Perceptron)**: Uma rede neural densamente conectada.
- **CNN 1D (Convolutional Neural Network)**: Uma rede neural convolucional unidimensional, adaptada para dados tabulares.
- **Híbrido CNN-LSTM**: Uma arquitetura que combina camadas convolucionais (CNN) para extração de características e camadas LSTM (Long Short-Term Memory) para capturar dependências sequenciais, adequada para dados que podem ter alguma ordem ou relação entre as características.

## Como Executar o Projeto

Siga os passos abaixo para configurar e executar o projeto:

### 1. Pré-requisitos
Certifique-se de ter o Python 3.10.0 instalado em seu sistema.

### 2. Clonar o Repositório (se aplicável)
```bash
git clone https://github.com/jabcneto/TCC_Redes_neurais_aplicadas_na_predicao_de_casos_de_diabetes.git
cd TCC_Redes_neurais_aplicadas_na_predicao_de_casos_de_diabetes
```

### 3. Instalar Dependências
Instale todas as bibliotecas Python necessárias usando o `pip`:
```bash
pip install -r requirements.txt
```

### 4. Estrutura de Dados
Certifique-se de que o arquivo `diabetes_prediction_dataset.csv` esteja na raiz do projeto ou no caminho especificado em `config.py`.

### 5. Executar o Pipeline
Para executar o pipeline completo de Machine Learning, basta rodar o script `main.py`:
```bash
python main.py
```

## Resultados

Após a execução do `main.py`, os resultados serão salvos no diretório `resultados_diabetes`, que incluirá:

- **`modelos/`**: Modelos treinados (arquivos `.keras` para Deep Learning e `.pkl` para modelos clássicos).
- **`graficos/`**: Diversos gráficos gerados durante a análise exploratória, avaliação e interpretabilidade (matrizes de confusão, curvas ROC, gráficos SHAP, explicações LIME).
- **`logs/`**: Arquivos de log detalhados da execução do pipeline.
- **`history/`**: Históricos de treinamento dos modelos Keras.
- Arquivos CSV com as métricas de avaliação de cada modelo e a comparação geral.

## Interpretabilidade

O projeto utiliza as bibliotecas SHAP e LIME para fornecer insights sobre como os modelos tomam suas decisões. Os gráficos e explicações gerados podem ser encontrados no diretório `resultados_diabetes/graficos/shap` e `resultados_diabetes/graficos/lime`.

## Contribuição

Sinta-se à vontade para contribuir com este projeto. Para isso, siga as diretrizes padrão de contribuição de código, como criar um fork, fazer suas alterações e enviar um Pull Request.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes. (Assumindo licença MIT, caso contrário, ajuste conforme necessário).

