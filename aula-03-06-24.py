#yuri rangel rabelo palheta

#aula 03-06-24

# Importações necessárias
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Digits
digitos = load_digits()

# Imagens dos dígitos
Images = digitos.images

# (Descomente para visualizar a primeira imagem)
# plt.matshow(Images[0])
# plt.gray()
# plt.show()

# Seleção das imagens e rótulos
X = digitos.images[:]
Y = digitos['target']

# Imprimir as formas das matrizes X e Y
print("Forma de X:", X.shape)
print("Forma de Y:", Y.shape)

# Filtragem dos dígitos 3, 5 e 9
ind2 = Y == 3
ind8 = Y == 9
ind5 = Y == 5

X2 = digitos.images[ind2]
X8 = digitos.images[ind8]
X5 = digitos.images[ind5]

Y2 = Y[ind2]
Y8 = Y[ind8]
Y5 = Y[ind5]

# (Descomente para visualizar exemplos dos dígitos filtrados)
# plt.matshow(X2[0])
# plt.gray()
# plt.show()

# plt.matshow(X8[0])
# plt.gray()
# plt.show()

# Concatenando as imagens e rótulos dos dígitos 3, 5 e 9
X = np.concatenate((X2, X8, X5), axis=0)
Y = np.concatenate((Y2, Y8, Y5), axis=0)

# Imprimir as novas formas das matrizes X e Y
print("Nova forma de X:", X.shape)
print("Nova forma de Y:", Y.shape)

# Remodelando as imagens para um formato plano (64 pixels por imagem)
X = X.reshape(len(X), 64)
print("Forma remodelada de X:", X.shape)

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print('Amostras de treinamento:', len(y_train))
print('Amostras de teste:', len(y_test))

# Treinamento e avaliação do classificador KNN
KNN1 = KNeighborsClassifier(n_neighbors=5)
KNN1.fit(X_train, y_train)
yp1 = KNN1.predict(X_test)
print("Previsões KNN:", yp1)
print("Valores reais:", y_test)
print("Acurácia KNN:", accuracy_score(y_test, yp1))

# Cálculo manual da acurácia
Ac = np.sum(1 * (yp1 == y_test)) / len(y_test)
print("Acurácia calculada manualmente:", Ac)

# Treinamento e avaliação do Perceptron
algoritmo = Perceptron()
algoritmo.fit(X_train, y_train)
yp2 = algoritmo.predict(X_test)
print("Previsões Perceptron:", yp2)
print("Valores reais:", y_test)
print("Acurácia Perceptron:", accuracy_score(y_test, yp2))

# Treinamento e avaliação da Regressão Logística
RL = LogisticRegression(random_state=0)
RL.fit(X_train, y_train)
yp3 = RL.predict(X_test)
print("Previsões Regressão Logística:", yp3)
print("Valores reais:", y_test)
print("Acurácia Regressão Logística:", accuracy_score(y_test, yp3))


'''
Comentários detalhados
Importações Necessárias:

numpy para manipulação de arrays.
load_digits para carregar o conjunto de dados dos dígitos.
matplotlib.pyplot para visualização dos dados.
train_test_split para dividir os dados em conjuntos de treinamento e teste.
KNeighborsClassifier, Perceptron, e LogisticRegression para os modelos de machine learning.
accuracy_score para calcular a acurácia dos modelos.
Carregamento e Visualização dos Dados:

Carregar o conjunto de dados dos dígitos e armazenar as imagens e rótulos.
Visualizar a forma dos dados e filtrar os dígitos específicos (3, 5 e 9).
Preparação dos Dados:

Concatenar os dados filtrados e remodelar as imagens para um formato plano (64 pixels por imagem).
Divisão dos Dados:

Dividir os dados em conjuntos de treinamento e teste.
Treinamento e Avaliação dos Modelos:

Treinar e avaliar três modelos diferentes: KNN, Perceptron, e Regressão Logística.
Calcular e imprimir a acurácia de cada modelo.
Esta estrutura fornece um fluxo claro desde o carregamento e preparação dos dados até o treinamento e avaliação dos modelos de machine learning.
'''