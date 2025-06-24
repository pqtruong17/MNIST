import numpy as np
import math
train = np.array([[1,1,1],[1,-1,1],[1,1,-1],[1,-1,-1]])
# train = np.array([[1,-1,1,-1],[1,1,-1,-1]])
result = np.array([1,-1,-1,-1])

lamda = 0.5
def softmax(array: np.ndarray):
    return np.exp(array)/ np.sum(np.exp(array))

def sigmoid(array: np.ndarray):
    return 1/(1+np.exp(-array))

def sigmoid_deriv(array: np.ndarray):
    return sigmoid(array)*(1 - sigmoid(array))

def init_param():
    weight1 = np.random.rand(4,3)
    weight2 = np.random.rand(4,4)
    weight3 = np.random.rand(4)
    return weight1, weight2, weight3

def prop(train: np.ndarray, weight1: np.ndarray, weight2: np.ndarray, weight3: np.ndarray, index: int):
    z1 = weight1.dot(train[index])
    a1 = sigmoid(z1)
    z2 = weight2.dot(a1)
    a2 = sigmoid(z2)
    z3 = weight3.dot(a2)
    delta3 = (result[index] - z3) * sigmoid_deriv(z3)
    dweight3 = a2.dot(delta3)
    delta2 = weight3.dot(delta3) * sigmoid_deriv(z2)
    dweight2 = a1.reshape(4,1).dot(delta2.reshape(1,4))
    delta1 = weight2.dot(delta2) * sigmoid_deriv(z1)
    dweight1 = delta1.reshape(4,1).dot(train[index].reshape(1,3))
    return dweight1, dweight2, dweight3

def update_param(weight1: np.ndarray, dweight1: np.ndarray, weight2: np.ndarray, dweight2: np.ndarray, weight3: np.ndarray, dweight3: np.ndarray):
    weight1 = weight1 + dweight1
    weight2 = weight2 + dweight2
    weight3 = weight3 + dweight3
    return weight1, weight2, weight3

def epoch(l):
    weight1, weight2, weight3 = init_param()
    for j in range(l):
        dweight1, dweight2, dweight3 = prop(train, weight1, weight2, weight3, 0)
        for index in range(3):
            d1, d2, d3 = prop(train, weight1, weight2, weight3, index + 1)
            dweight1 += d1
            dweight2 += d2
            dweight3 += d3
        weight1, weight2, weight3 = update_param(weight1, dweight1, weight2, dweight2, weight3, dweight3)
    return weight1, weight2, weight3
for i in range(25):
    weight1, weight2, weight3= epoch(100)
    for i in range(4):
        z1 = weight1.dot(train[i])
        a1 = sigmoid(z1)
        z2 = weight2.dot(a1)
        a2 = sigmoid(z2)
        z3 = weight3.dot(a2)
        print(z3)
    print("\n")
