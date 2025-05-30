import numpy as np
import time
import sys
import os

def sample(type: str):
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")
    with open(f"../dataset/{type}.npy", "rb") as fp:
        with open(f"./dataset/{type}.npy", "wb") as fpp:
            for i in range(10):
                value = np.load(fp)
                value[:, 1:785][value[:, 1:785] > 0] = 1
                np.save(fpp, value)

def weights():
    if not os.path.exists("./weights"):
        os.mkdir("./weights")
    with open("./dataset/train.npy", "rb") as fp:
        with open("./weights/weights.npy", "wb") as fpp:
            for i in range(10):
                train = np.load(fp)
                array = np.sum(train, axis=0) / len(train)
                np.save(fpp, array)

def bayesian(name: str):
    array = list()
    with open(f"./weights/weights.npy", "rb") as fp:
        for i in range(10):
            value = np.load(fp)
            array.append(value)
    with open(f"./dataset/test.npy", "rb") as fpp:
        fin = []
        for k in range(10):
            test = np.load(fpp)
            res = [0]*10
            for j in range(len(test)):
                sol = list()
                for i in range(10):
                    temp = array[i].copy()
                    temp[1:785][test[j][1:785] == 0] = 1 - temp[1:785][test[j][1:785] == 0]
                    sol.append(np.prod(temp[1:785]))
                res[np.argmax(sol)] += 1
            fin.append(np.divide(res,len(test)/100))
    with open(f"{name}.npy", "wb") as fppp:
        np.save(fppp, fin)

def main(args):
    startTime = time.time()
    sample("test")
    sample("train")
    weights()
    bayesian("matrix")
    endTime = time.time()
    print(f"{startTime-endTime}")

if __name__ == "__main__":
    main(sys.argv[1:])
