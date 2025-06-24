import numpy as np
import math

train = np.array([[0,0],[0,1],[1,0],[1,1]])
# train = np.array([[1,-1,1,-1],[1,1,-1,-1]])
result = np.array([[0,1,1,0]])
# result = np.array([[1,0],[0,1],[0,1],[0,1]])


def softmax(array: np.ndarray):
    return np.exp(array)/ np.sum(np.exp(array))

def sigmoid(array: np.ndarray):
    return 1/(1+np.exp(-array))

def sigmoid_deriv(array: np.ndarray):
    return sigmoid(array)*(1 - sigmoid(array))

def init_param(n):
    weight1 = np.random.rand(n,2)
    weight2 = np.random.rand(1,n)
    return weight1, weight2

def Relu(array: np.ndarray):
    return np.maximum(array,0)

def Relu_deriv(array: np.ndarray):
    return array > 0

def prop(train: np.ndarray, weight1: np.ndarray, weight2: np.ndarray):
    z1 = weight1.dot(train.T)
    a1 = sigmoid(z1)
    z2 = weight2.dot(a1)
    delta2 = (result - z2) * sigmoid_deriv(z2)
    dweight2 = delta2.dot(a1)
    delta1 = weight2.T.dot(delta2) * sigmoid_deriv(z1)
    dweight1 = delta1.dot(train)
    return dweight1, dweight2

def update_param(weight1: np.ndarray, dweight1: np.ndarray, weight2: np.ndarray, dweight2: np.ndarray):
    weight1 = weight1 + dweight1
    weight2 = weight2 + dweight2
    return weight1, weight2

def epoch(l):
    weight1, weight2 = init_param(4)
    for j in range(l):
        dweight1, dweight2, = prop(train, weight1, weight2)
        weight1, weight2, = update_param(weight1, dweight1, weight2, dweight2)
    return weight1, weight2

for i in range(25):
    weight1, weight2 = epoch(500)
    z1 = weight1.dot(train.T)
    a1 = sigmoid(z1)
    z2 = weight2.dot(a1)
    print(z2)
    print("\n")
