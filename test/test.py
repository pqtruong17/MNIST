import numpy as np

# x = np.array([[0,0],[0,1],[1,0],[1,1]])
x = np.array([[0,0,1,1],[0,1,0,1]])
y = np.array([0,1,1,0])

def Relu_deriv(array: np.ndarray):
    array[array > 0] = 1
    array[array <= 0] = 0
    return array

def Relu(array: np.ndarray):
    return np.maximum(array, 0)

def sigmoid(array: np.ndarray):
    return 1/(1+np.exp(-array))

def softmax(array: np.ndarray): 
    return np.exp(array)/np.sum(array)

W = np.random.rand(2,2)
c = np.random.rand(2,)
w = np.random.rand(2,)
b = np.random.rand(4,)

hidden = W@x + np.array([c]*4).T
output = w@hidden + b
g = y - output
g = g * output 
W = hidden@g


print(output.shape)
print(hidden.shape)
print(W)
