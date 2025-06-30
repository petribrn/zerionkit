## ZerionKit
Projeto final INE5664

### Alunos:

- Bruno Petri (21106217)
- Carolina Pacheco da Silva (21104291)
- Franco Saravia Tavares (21106477)

### Visão geral da implementação

A implementação consiste em uma Rede Neural (RNA) para treinamento de modelos de classificação binária, multiclasse e regressão,
aplicando o algoritmo de backpropapagation (fases feedforward + backpropagation).

### Estrutura de pastas

- `src/configs/constants.py`: Constantes auxiliares para reuso de strings nos arquivos do projeto;
- `src/data_handler.py`: Lógica de tratamento de dados para o cliente da RNA. É realizado o processamento de datasets
importados em arquivos .csv da pasta `data` e a divisão deles em datasets de treinamento, validação e teste, em cada um deles
em inputs (x) e target outputs (y);
- `src/layer.py`: Construtor de uma camada da RNA, contendo atributos como tamanho (em nro de neurônios), 
função de ativação (**Linear**, **Sigmoid**, **ReLU** e **Softmax**) e derivada da função de ativação correspondente;
- `src/loss.py`: Implementação das funções de perda **Square Error**, **Binary Cross Entropy**, **Cross Entropy**;
- `src/zerion_nn.py`: Implementação da RNA, contendo a geração aleatória dos parâmetros iniciais (weights e bias),
feedforward, cálculo dos gradientes de erros da saída, backpropagation dos gradientes, e impl. do algoritmo do 
gradiente descendente para atualização dos parâmetros. Também contém métodos para os fluxos de treinamento (`train()`)
e teste (`evaluate()`) da rede;
- `examples/binary_class.py`: Código cliente da RNA de exemplo para o problema de **classificação binária**, contendo a
instanciação do data handler e da RNA para treinamento, validação e teste;
- `examples/multi_class.py`: Código cliente da RNA de exemplo para o problema de **classificação multiclasse**, contendo a
instanciação do data handler e da RNA para treinamento, validação e teste;
- `examples/regression.py`: Código cliente da RNA de exemplo para o problema de **regressão**, contendo a
instanciação do data handler e da RNA para treinamento, validação e teste;
- `main.py`: Código cliente geral da RNA, contendo a instanciação do data handler e da RNA para treinamento, validação e teste.

### Passos para execução

1. Tenha o [Python](https://www.python.org/) na versão 3.12.2 instalado no sistema.
    Uma sugestão de gerenciamento de versões do Python, é o [Pyenv](https://github.com/pyenv/pyenv).

2. Ideal que utilize um venv (ambiente virtual do Python) para instalar as dependências, rode, na raiz do projeto:
   - `pip install requirements.txt`

3. Rode no terminal, estando na pasta raiz do projeto:
   - `python3 main.py regression`, para testar o exemplo de regressão;
   - `python3 main.py binary_class`, para testar o exemplo de classificação binária;
   - `python3 main.py multi_class`, para testar o exemplo de classificação multiclasse;
