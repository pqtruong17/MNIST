import math, sys
import numpy as np
import matplotlib.pyplot as plt

identity = np.identity(10)


class network:
    def __init__(self, n = 30):
        self.size = 0
        self.lr = 1
        self.al = 0.01
        self.weight1 = (np.random.rand(n,784) - 0.5) * np.sqrt(2/(n+784))
        self.bias1 = np.random.rand(n,1)
        self.weight2 = (np.random.rand(10,n) - 0.5) * np.sqrt(2/(10+n))
        self.bias2 = np.random.rand(10,1)
        self.percent = 0
        self.lengths = []
        self.matrix = np.identity(10)
        self.array, self.train, self.Y, self.result = self.setup()

    def setup(self, name = "train"):
        identity = np.identity(10)
        X = np.array([])
        self.size = 0
        self.lengths = []
        with open(f"../dataset/{name}.npy", "rb") as fp:
            for digit in range(10):
                value = np.load(fp)
                self.lengths.append(len(value))
                X = np.append(X,value)
        self.size = np.sum(self.lengths)
        X = X.reshape(self.size,785)
        Y = np.array([int(d[0]) for d in X])
        array = X
        result = np.array([identity[value] for value in Y])
        train = X[:, 1:785] / 255
        return array, train, Y, result

    def shuffle(self, size = 10000, check = True):
        if check:
            np.random.shuffle(self.array)
        self.size = size
        self.array = self.array[:size, :]
        self.Y = np.array([int(value[0]) for value in self.array])
        self.result = np.array([identity[value] for value in self.Y])
        self.train = self.array[:, 1:785] / 255

    def sigmoid(self, array):
        array = np.clip(array, -500, 500)
        return 1/(1+np.exp(-array))

    def sigmoid_deriv(self, array):
        return self.sigmoid(array)*(1 - self.sigmoid(array))

    def foward(self, X):
        z1 = self.weight1 @ X + self.bias1
        a1 = self.sigmoid(z1)
        z2 = self.weight2 @ a1 + self.bias2
        return z1, a1, z2

    def backward(self, z1, a1, z2):
        delta2 = (self.result.T - z2) * self.sigmoid_deriv(z2)
        dbias2 = np.sum(delta2, axis=1, keepdims=True)
        dweight2 = delta2 @ a1.T + self.al * np.sign(self.weight2)
        delta1 = self.weight2.T @ delta2 * self.sigmoid_deriv(z1)
        dbias1 = np.sum(delta1, axis=1, keepdims=True)
        dweight1 = delta1 @ self.train + self.al * np.sign(self.weight1)
        return dweight1, dbias1, dweight2, dbias2

    def update_weight(self, dweight1, dbias1, dweight2, dbias2):
        self.weight2 = self.weight2 + self.lr * dweight2/self.train.shape[0] 
        self.bias2 = self.bias2 + self.lr * dbias2/self.train.shape[0]
        self.weight1 = self.weight1 + self.lr * dweight1/self.train.shape[0] 
        self.bias1 = self.bias1 + self.lr * dbias1/self.train.shape[0]

    def epoch(self,l = 1000):
        self.shuffle()
        for j in range(l):
            z1, a1, z2 = self.foward(self.train.T)
            dweight1, dbias1, dweight2, dbias2 = self.backward(z1, a1, z2)
            self.update_weight(dweight1, dbias1, dweight2, dbias2)

    def prediction(self):
        array, train, Y, result = self.setup("test")
        z1, a1, z2 = self.foward(train.T)
        predictions = np.argmax(z2, axis = 0)
        self.percent = np.sum(predictions == Y)/self.size
        predictions = np.argmax(z2, axis=0)
        arr = np.array([], dtype=int)
        s = 0
        for i in range(10):
            temp = predictions[s : s + self.lengths[i]]
            s += self.lengths[i]
            value = np.bincount(temp)
            arr = np.append( arr, np.pad(value, (0, 10 - len(value)), "constant")/self.lengths[i])
        self.matrix = arr.reshape(10,10) * 100

    def confusion_matrix(self):
        labels = [f"{i}" for i in range(np.shape(self.matrix)[0])]
        fig, ax = plt.subplots(figsize=(10, 10))  # type:ignore
        ax.imshow(self.matrix, cmap="viridis")
        for i in range(np.shape(self.matrix)[0]):
            for j in range(np.shape(self.matrix)[1]):
                ax.text(j, i, f"{self.matrix[i, j]:.2f}", ha="center", va="center", color="white")
        ax.set_xticks(np.arange(np.shape(self.matrix)[1]))
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(np.shape(self.matrix)[0]))
        ax.set_yticklabels(labels)
        ax.xaxis.set_ticks_position("top")
        plt.tight_layout()
        fig.savefig(f"./matrix.png", bbox_inches="tight", pad_inches=1)

n = network()
n.epoch(5000)
n.prediction()
print(n.percent)
print(n.matrix)
n.confusion_matrix()
