import numpy as np
from numpy import linalg as LA

# See README.md for more information about the KFOLDXV(D,A,L,k) algorithmn


def k_neighbor(k: int):
    # D, the given dataset, with elements z^(i) (digit 0-9)
    # from both the test and train data.
    with open("./dataset/train.npy", "rb") as f:
        train = np.load(f)
    with open("./dataset/test.npy", "rb") as f:
        test = np.load(f)
    matrix = np.arange(1000).reshape(10, 100)
    ans = np.arange(0)
    for line in matrix:
        # initilalize e the cross examination result between the train and test data
        temp = np.zeros(10, dtype=int)
        for index in line:
            # e_j = L(f_i,z^(j)) is error for each train digit j
            distance = LA.norm(
                # f_i = A(D\D_i) is the digit test dataset
                np.array([test[index]] * 1000) -
                # z^(j) is the digit train dataset in D_i
                np.array(train), axis=1)
            # Split D into k exclusive subset D_i, whose union is D
            partition = np.argpartition(distance, k)[:k]
            # D, the given dataset, with elements z^(i), the digit train dataset
            temp += np.bincount(partition // 100, minlength=len(temp))
        ans = np.append(ans, temp)
    # return e the cross examination between the train and test data
    return ans.reshape(10, 10)
