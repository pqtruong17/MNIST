import numpy as np
import os
import sys


def sample(type: str):
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")
    with open(f"../dataset/{type}.npy", "rb") as f:
        data = np.arange(0)
        for i in range(10):
            value = np.load(f)
            rand = np.random.choice(len(value), 100)
            data = np.append(data, value[rand])
        with open(f"./dataset/{type}.npy", "wb") as f:
            np.save(f, data.reshape(len(data) // 785, 785))


def main(args):
    for arg in args:
        if arg == "train" or arg == "test":
            sample(arg)


if __name__ == "__main__":
    main(sys.argv[1:])
