#aula 29-05-24
#yuri rangel rabelo palheta

#decomposição de valores singulares
#utiliza o minimo de dados

#singular value decomposition (svd)
#import numpy as np

#X=np.random.rand(5,3)

#U, S, V = np.linalg.svd(X)

from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


caminho = 'c:/Users/yurim/OneDrive/Área de Trabalho/Apr-Maq/dog.jpg'

X = imread(caminho)

np.shape(X)

plt.imshow(X)

plt.show()

X = np.mean(X, -1)

np.shape(X)

img = plt.imshow(X)

img.set_cmap('gray') #o cachorro fica verde se nao mapear como gray

plt.show()


U, S, V = np.linalg.svd(X , full_matrices=False)

Sn=S/np.sum(S)

plt.semilogy(S)
plt.grid()
plt.show()

Sd = np.diag(S)

#redesenha o cachorro
r1=200

Xr1 = U[:,0:r1] @ Sd[0:r1,0:r1] @ V[0:r1,:]

img = plt.imshow(Xr1)

img.set_cmap('gray')

plt.show()


Sn=S/np.sum(S)

plt.subplot(121)
plt.semilogy(Sn)
plt.grid()
plt.show()
plt.subplot(122)
plt.plot(np.cumsum(Sn)[0:60])
plt.grid()
plt.show()

######################################

#yuri rangel rabelo palheta

#aula 29-05-24-2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Caminhos para os arquivos
caminho_y = 'c:/Users/yurim/OneDrive/Área de Trabalho/Apr-Maq/ovariancancer_grp.csv'
caminho_x = 'c:/Users/yurim/OneDrive/Área de Trabalho/Apr-Maq/ovariancancer_obs.csv'

# Carregar dados
y = pd.read_csv(caminho_y).values[:,0]
x = pd.read_csv(caminho_x).values

# Convertendo rótulos para numéricos
y[y=='Cancer'] = 1
y[y=='Normal'] = 0
y = y.astype(int)  # Certificando que y é do tipo inteiro

# Normalizando os dados
Xm = np.mean(x, 0)
Xn = x - Xm

# Decomposição SVD
U, S, V = np.linalg.svd(Xn, full_matrices=False)

# Normalizando os valores singulares
Sn = S/np.sum(S)

# Plotagem dos valores singulares
plt.semilogy(Sn[:-1])
plt.semilogy(Sn[:-1], '*r')
plt.grid()
plt.show()

# Plotagem 2D de U
plt.scatter(U[:,0], U[:,1], c=y)
plt.grid()
plt.show()

# Plotagem 2D das observações originais
plt.scatter(x[:,0], x[:,1], c=y)
plt.grid()
plt.show()

# Plotagem 3D de U com rotação
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(U[:, 0], U[:, 1], U[:, 2], c=y)
ax.set_xlabel('U[:,0]')
ax.set_ylabel('U[:,1]')
ax.set_zlabel('U[:,2]')
plt.colorbar(sc)
plt.grid()

# Definindo ângulos de visualização para melhor separação
angles = [(30, 30), (60, 30), (90, 30), (120, 30)]

for angle in angles:
    ax.view_init(elev=angle[0], azim=angle[1])
    plt.draw()
    plt.pause(1)

plt.show()