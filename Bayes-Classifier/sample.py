import os
import numpy as np
import sys


def sample(type: str):
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")
    with open(f"../dataset/{type}.npy", "rb") as fp:
        with open(f"./dataset/{type}.npy", "wb") as fpp:
            for i in range(10):
                value = np.load(fp)
                value[:, 1:785][value[:, 1:785] > 0] = 1
                np.save(fpp, value)


def main(args):
    for arg in args:
        if arg == "train" or arg == "test":
            sample(arg)


if __name__ == "__main__":
    main(sys.argv[1:])
