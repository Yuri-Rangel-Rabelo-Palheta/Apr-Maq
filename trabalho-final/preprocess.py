import pandas as pd
import numpy as np
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing import image
#from tensorflow.keras_preprocessing.image import ImageDataGenerator
#from tensorflow.keras_preprocessing import image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Carregar as anotações do dataset CelebA
attributes = pd.read_csv('C:/Users/yurim/OneDrive/Área de Trabalho/Apr-Maq/trabalho-final/arquivo/list_attr_celeba.csv')
images_path = 'C:/Users/yurim/OneDrive/Área de Trabalho/Apr-Maq/trabalho-final/arquivo/img_align_celeba/img_align_celeba/'

# Visualizar algumas amostras de atributos
print(attributes.head())

# Carregar e processar as imagens
def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for img_path in image_paths:
        img = image.load_img(images_path + img_path, target_size=target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalização
        images.append(img)
    return np.vstack(images)

# Exemplo de carregamento de um subset de dados
image_paths = attributes['image_id'].head(1000)
X = load_and_preprocess_images(image_paths)
y = attributes.head(1000).drop(columns=['image_id'])

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Visualização de algumas imagens
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i])
    plt.title(y_train.iloc[i].idxmax())
    plt.axis('off')
plt.show()
