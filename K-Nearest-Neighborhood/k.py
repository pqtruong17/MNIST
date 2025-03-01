import numpy as np
from numpy import linalg as LA


def k_neighbor(k: int):
    with open("./dataset/train.npy", "rb") as f:
        train = np.load(f)
    with open("./dataset/test.npy", "rb") as f:
        test = np.load(f)
    matrix = np.arange(1000).reshape(10, 100)
    ans = np.arange(0)
    for line in matrix:
        temp = np.zeros(10, dtype=int)
        for index in line:
            distance = LA.norm(np.array([test[index]] * 1000) - np.array(train), axis=1)
            partition = np.argpartition(distance, k)[:k]
            temp += np.bincount(partition // 100, minlength=len(temp))
        ans = np.append(ans, temp)
    return ans.reshape(10, 10)

print(k_neighbor(1))
