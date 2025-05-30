import numpy as np

with open("./dataset/train.npy", "rb") as fp:
    for i in range(10):
        test =  np.load(fp)
        print(len(test))
