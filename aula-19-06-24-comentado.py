#yuri rangel rabelo palheta
#aula 19-06-24

#LDA - Linear Discriminant Analysis

#LDA é o nome do algoritmo que faz o processo de redução de dimensionalidade
#para que os dados sejam mais parecidos entre si. 
#O processo de redução de dimensionalidade consiste na redução das dimensoes do conjunto de dados para que
#os dados sejam mais parecidos entre si.

from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score

# Gera um conjunto de dados sintético
X, y = make_classification(n_samples=300, 
                           n_features=2, 
                           n_informative=2, 
                           n_redundant=0, 
                           random_state=3)

# Plota os dados gerados em um gráfico de dispersão
plt.scatter(X[:,0], X[:,1], c=y, cmap='rainbow')
plt.grid()
plt.show()

# Divide o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Cria uma instância do LDA e ajusta o modelo aos dados de treinamento
LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train, y_train)

# Faz previsões nos dados de teste
yp = LDA.predict(X_test)

# Calcula e imprime a acurácia do modelo
print(accuracy_score(y_test, yp))

# Cria e imprime a matriz de confusão
Mc = confusion_matrix(y_test, yp)
print(Mc)

# Plota a matriz de confusão usando seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(Mc, annot=True, cmap='Blues', fmt='g', cbar=False, annot_kws={'size': 14}, linewidths=0.5, linecolor='grey')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.xticks(ticks=[0.5, 1.5], labels=['Negativo', 'Positivo'])
plt.yticks(ticks=[0.5, 1.5], labels=['Negativo', 'Positivo'])
plt.show()

# Executa validação cruzada com 5 dobras e calcula a acurácia média
svc = cross_val_score(LDA, X, y, cv=5, scoring='accuracy')
print(svc)
print(np.mean(svc))


"""
Importações e Geração de Dados:

make_classification é usado para criar um conjunto de dados sintético com 300 amostras, 2 características informativas e sem características redundantes.
plt.scatter é usado para plotar os dados gerados, coloridos de acordo com suas classes.
Divisão dos Dados:

train_test_split divide os dados em conjuntos de treinamento e teste, com 60% dos dados para treinamento e 40% para teste.
Ajuste do Modelo:

LinearDiscriminantAnalysis é instanciado e ajustado aos dados de treinamento usando LDA.fit(X_train, y_train).
Previsão e Avaliação:

LDA.predict(X_test) faz previsões nos dados de teste.
accuracy_score(y_test, yp) calcula a acurácia das previsões.
confusion_matrix(y_test, yp) cria uma matriz de confusão que é plotada usando seaborn.
Validação Cruzada:

cross_val_score é usado para realizar validação cruzada com 5 dobras e calcular a acurácia média.
Resultados:

O gráfico de dispersão inicial mostra a distribuição dos dados.
A acurácia impressa é a proporção de previsões corretas feitas pelo modelo no conjunto de teste.
A matriz de confusão mostra a contagem de previsões verdadeiras e falsas para cada classe.
O gráfico de calor da matriz de confusão facilita a visualização da performance do modelo.
A validação cruzada fornece a acurácia média do modelo em diferentes subconjuntos dos dados, oferecendo uma avaliação mais robusta da performance do modelo.
Este código mostra como usar LDA para classificação e avaliar a performance do modelo usando várias métricas.
"""