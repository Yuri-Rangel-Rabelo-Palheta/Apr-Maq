#Yuri Rangel Rabelo Palheta
#Aula 03-07-24

import numpy as np
import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.utils import to_categorical
#from keras.datasets import mnist

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, Activation
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.datasets import mnist

# Carregando o dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Fixando a semente aleatória
np.random.seed(42)

# Selecionando 9 imagens aleatórias para visualização
ind = np.random.randint(0, X_train.shape[0], size=9)
images = X_train[ind]

# Plotando as imagens
plt.figure(figsize=(5, 5))
for i in range(len(ind)):
    plt.subplot(3, 3, i + 1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
plt.show()

# Convertendo os labels para one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Normalizando e formatando os dados de entrada
image_size = X_train.shape[1]
input_size = image_size * image_size
X_train = X_train.reshape(-1, input_size)
X_train = X_train.astype('float32') / 255

X_test = X_test.reshape(-1, input_size)
X_test = X_test.astype('float32') / 255

# Parâmetros da RNA
batch_size = 128
hidden_units = 100
dropout = 0.45
num_labels = 10

# Construção do modelo
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Resumo do modelo
model.summary()

# Compilação e treinamento do modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=batch_size)


model.evaluate(X_test, y_test, batch_size=batch_size)

print('Final Accuracy: %.2f%%' % (100 * model.evaluate(X_test, y_test, batch_size=batch_size)[1]))

HistoricoTreino = model.fit(X_train, y_train, epochs=30, batch_size=batch_size)

plt.plot(HistoricoTreino.history['accuracy'])
plt.title('Acurácia do modelo')
plt.ylabel('Acurácia')
plt.xlabel('Epoques')
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.show()
