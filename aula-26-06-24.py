#aula 26-06-24
#yuri rangel rabelo palheta

# SVM - Support Vector Machines - resolvendo o iris dataset

from sklearn import datasets

# Carregando o conjunto de dados Iris
iris = datasets.load_iris()

# X são as características do conjunto de dados (selecionando as duas primeiras características)
X = iris.data[:, 0:2]

# y são os rótulos das classes
y = iris.target

# classes contém os nomes das classes
classes = iris.target_names

# Importando biblioteca para visualização
import matplotlib.pyplot as plt

# Criando um gráfico de dispersão das duas primeiras características
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.grid()
plt.show()

# Sugestão do professor
# X -> PCA -> X' -> Classificador (KNN,SVN, ...)

# Dividindo os dados em conjuntos de treino e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Importando a biblioteca para SVM
from sklearn import svm

# Criando modelos SVM com diferentes kernels
SVM_linear = svm.SVC(kernel='linear')  # Kernel linear
SVM_poly = svm.SVC(kernel='poly', degree=3)  # Kernel polinomial de grau 3
SVM_rbf = svm.SVC(kernel='rbf', gamma=0.4)  # Kernel RBF com gamma 0.4

# Sugestão do professor: gridsearch para otimizar os parâmetros dos modelos

# Treinando os modelos SVM com os dados de treino
SVM_linear.fit(X_train, y_train)
SVM_poly.fit(X_train, y_train)
SVM_rbf.fit(X_train, y_train)

# Fazendo previsões nos dados de teste
yp_linear = SVM_linear.predict(X_test)
yp_poly = SVM_poly.predict(X_test)
yp_rbf = SVM_rbf.predict(X_test)

# Importando função para calcular a matriz de confusão
from sklearn.metrics import confusion_matrix

# Imprimindo a matriz de confusão para cada modelo
print("Confusion Matrix - Linear Kernel")
print(confusion_matrix(y_test, yp_linear))

print("Confusion Matrix - Polynomial Kernel")
print(confusion_matrix(y_test, yp_poly))

print("Confusion Matrix - RBF Kernel")
print(confusion_matrix(y_test, yp_rbf))

