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

X, y = make_classification(n_samples=300, 
                           n_features=2, 
                           n_informative=2, 
                           n_redundant=0, 
                           random_state=3)


#print(X.shape, y.shape)


plt.scatter(X[:,0], X[:,1], c=y, cmap='rainbow')
plt.grid()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

LDA = LinearDiscriminantAnalysis()

LDA.fit(X_train, y_train)

yp = LDA.predict(X_test)

print(accuracy_score(y_test, yp))

#******************8

Mc = confusion_matrix(y_test, yp)

print(Mc)

#plot =plt.matshow(Mc, cmap='rainbow')
#plt.colorbar(plot)
#plt.show()

import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(Mc, annot=True, cmap='Blues', fmt='g', cbar=False, annot_kws={'size': 14}, linewidths=0.5, linecolor='grey')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.xticks(ticks=[0.5, 1.5], labels=['Negativo', 'Positivo'])
plt.yticks(ticks=[0.5, 1.5], labels=['Negativo', 'Positivo'])
plt.show()


from sklearn.model_selection import cross_val_score


LDA = LinearDiscriminantAnalysis()

svc = cross_val_score(LDA, X, y, cv=5, scoring='accuracy')

print(cross_val_score(LDA, X, y, cv=5, scoring='accuracy'))


import numpy as np


print(np.mean(svc))

