# Projeto de Predição de Diabetes com Redes Neurais e Modelos Clássicos

Trabalho de Conclusão de Curso (TCC) apresentado ao curso de Ciência da Computação da Universidade Católica de Petrópolis (UCP).

Este projeto tem como objetivo desenvolver, treinar e comparar modelos de machine learning para predição de diabetes, utilizando tanto redes neurais (MLP, CNN, híbrido) quanto algoritmos clássicos.

## Estrutura do Projeto

- `diabetes_prediction_dataset.csv`: Base de dados utilizada para treinamento e teste dos modelos.
- `rede_neural_melhorada.ipynb`: Notebook principal com todo o pipeline de pré-processamento, treinamento, avaliação e comparação dos modelos.
- `resultados_diabetes/`: Resultados, métricas, relatórios e gráficos gerados durante os experimentos.
    - `graficos/`: Gráficos de desempenho, curvas ROC, matrizes de confusão, importância de variáveis, etc.
    - `history/`: Histórico de treinamento dos modelos de redes neurais.
    - `logs/`: Logs do TensorBoard para visualização do treinamento.
    - `modelos/`: Modelos treinados salvos em formato `.h5`.

## Como Executar

1. **Pré-requisitos:**
   - Python 3.8+
   - Instalar dependências:
     ```bash
     pip install -r requirements.txt
     ```
2. **Executar o notebook:**
   - Abra o `rede_neural_melhorada.ipynb` no Jupyter Notebook ou JupyterLab.
   - Execute as células sequencialmente.

## Principais Modelos Utilizados
- **Redes Neurais:**
  - MLP (Perceptron Multicamadas)
  - CNN (Rede Neural Convolucional)
  - Modelo Híbrido (MLP + CNN)
- **Modelos Clássicos:**
  - Random Forest
  - SVM
  - Regressão Logística
  - KNN
  - Outros

## Resultados
Os resultados das avaliações dos modelos estão disponíveis na pasta `resultados_diabetes/` em arquivos `.csv` e gráficos.

## Visualização
Para visualizar o treinamento das redes neurais, utilize o TensorBoard apontando para a pasta `resultados_diabetes/logs/`:
```bash
 tensorboard --logdir resultados_diabetes/logs/
```

## Autor
- João Antonio Barcelos Coutinho Neto

## Licença
Este projeto é apenas para fins acadêmicos.
