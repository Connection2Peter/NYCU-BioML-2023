import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def VisualVector3D(X, y):
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(X)
    lenX = len(X)
    PositiveData = [reduced_data[i] for i in range(lenX) if y[i] == 1]
    NegativeData = [reduced_data[i] for i in range(lenX) if y[i] == 0]
    
    PositiveData = random.sample(PositiveData, int(len(PositiveData)))
    NegativeData = random.sample(NegativeData, int(len(NegativeData)/10))
    
    print(np.shape(PositiveData), np.shape(NegativeData))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*zip(*PositiveData), c='b', marker='o', label='Positive')
    ax.scatter(*zip(*NegativeData), c='r', marker='x', label='Negative')

    ax.set_title('3D High-Dimensional Vector Distribution')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    ax.legend()
    
    plt.show()
