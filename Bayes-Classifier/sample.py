import numpy as np

trains = list()
tests = list()
with open("../dataset/train.npy", "rb") as fp:
    k = 100
    for i in range(10):
        train = np.load(fp)
        tests = np.delete(train, range(0,k,1))
        trains.append(train)
