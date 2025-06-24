import math, sys
import numpy as np

import matplotlib.pyplot as plt


train = np.array([])

identity = np.identity(10)
size = 100
lr = 2

with open("../dataset/train.npy", "rb") as fp:
    for digit in range(10):
        value = np.load(fp)
        train = np.append(train,value[0:size])


train = train.reshape(size*10,785)
np.random.shuffle(train)
Y = np.array([int(value[0]) for value in train])
result = np.array([identity[value] for value in Y])
train = train[:, 1:785] / 255


def softmax(array: np.ndarray):
    shift_array = array - np.max(array,axis=1, keepdims=True)
    exp_vals = np.exp(shift_array)
    return exp_vals/ np.sum(exp_vals, axis=1, keepdims=True)

def sigmoid(array: np.ndarray):
    array = np.clip(array, -500, 500)
    return 1/(1+np.exp(-array))

def sigmoid_deriv(array: np.ndarray):
    return sigmoid(array)*(1 - sigmoid(array))

def init_param(n: int):
    weight1 = (np.random.rand(n,784) - 0.5) * np.sqrt(2/(n+784))
    bias1 = np.random.rand(n,1)
    weight2 = (np.random.rand(10,n) - 0.5) * np.sqrt(2/(10+n))
    bias2 = np.random.rand(10,1)
    return weight1, bias1, weight2, bias2

def Relu(array: np.ndarray):
    return np.maximum(array,0)

def Relu_deriv(array: np.ndarray):
    return array > 0

def prop(train: np.ndarray, weight1: np.ndarray, bias1: np.ndarray, weight2: np.ndarray, bias2: np.ndarray):
    z1 = weight1 @ train.T + bias1
    a1 = sigmoid(z1)
    z2 = weight2 @ a1 + bias2
    # a2 = softmax(z2)
    delta2 = (result.T - z2) * sigmoid_deriv(z2)
    dbias2 = np.sum(delta2, axis=1, keepdims=True)
    dweight2 = delta2 @ a1.T
    delta1 = weight2.T @ delta2 * sigmoid_deriv(z1)
    dbias1 = np.sum(delta1, axis=1, keepdims=True)
    dweight1 = delta1 @ train
    return dweight1, dbias1, dweight2, dbias2

def update_weight(weight1: np.ndarray, dweight1: np.ndarray, weight2: np.ndarray, dweight2: np.ndarray):
    weight1 = weight1 + lr * dweight1/train.shape[0]
    weight2 = weight2 + lr * dweight2/train.shape[0]
    return weight1, weight2 

def update_bias(bias1: np.ndarray, dbias1: np.ndarray, bias2: np.ndarray, dbias2: np.ndarray):
    bias1 = bias1 + lr * dbias1/train.shape[0]
    bias2 = bias2 + lr * dbias2/train.shape[0]
    return bias1, bias2 

def epoch(l):
    weight1, bias1, weight2, bias2 = init_param(30)
    for j in range(l):
        dweight1,dbias1, dweight2, dbias2 = prop(np.array(train), weight1, bias1, weight2, bias2)
        weight1, weight2 = update_weight(weight1, dweight1, weight2, dweight2)
        bias1, bias2 = update_bias(bias1, dbias1, bias2, dbias2)
    return weight1, bias1, weight2, bias2



def confusion_matrix(arr):
    labels = [f"{i}" for i in range(np.shape(arr)[0])]
    fig, ax = plt.subplots(figsize=(10, 10))  # type:ignore
    ax.imshow(arr, cmap="viridis")
    for i in range(np.shape(arr)[0]):
        for j in range(np.shape(arr)[1]):
            ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", color="white")
    ax.set_xticks(np.arange(np.shape(arr)[1]))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(np.shape(arr)[0]))
    ax.set_yticklabels(labels)
    ax.xaxis.set_ticks_position("top")
    plt.tight_layout()
    fig.savefig(f"./confusion_matrix.png", bbox_inches="tight", pad_inches=1)

w1, b1, w2, b2 = epoch(1000)

train = np.array([])
with open("../dataset/test.npy", "rb") as fp:
    for digit in range(10):
        value = np.load(fp)
        train = np.append(train,value[0:size])
train = train.reshape(size*10,785)

# np.random.shuffle(train)
Y = np.array([int(value[0]) for value in train])
result = np.array([identity[value] for value in Y])
train = train[:, 1:785] / 255

z1 = w1 @ train.T + b1
a1 = sigmoid(z1)
z2 = w2 @ a1 + b2
predictions = np.argmax(z2, axis = 0)
print(np.sum(predictions == Y)/(size*10))
predictions = np.argmax(z2, axis=0).reshape(10,size)
arr = np.array([])
for value in predictions:
    value = np.array(value)
    temp = np.array([np.sum(value == 0), np.sum(value == 1), np.sum(value == 2), np.sum(value == 3), np.sum(value == 4), np.sum(value == 5), np.sum(value == 6), np.sum(value == 7), np.sum(value == 8), np.sum(value == 9)])
    arr = np.append(arr, temp)

arr = arr.reshape(10,10)
print(arr)
confusion_matrix(arr)
# matrix(arr, "confusion")
# print("Predicted classes:\n", predictions.reshape(10,size))

