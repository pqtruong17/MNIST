import numpy as np
import sys
import os


def weights():
    if not os.path.exists("./weights"):
        os.mkdir("./weights")
    with open("./dataset/train.npy", "rb") as fp:
        with open("./weights/weights.npy", "wb") as fpp:
            for i in range(10):
                train = np.load(fp)
                np.save(fpp, np.sum(train, axis=0) / len(train))


def main(args):
    for arg in args:
        if arg == "weights":
            weights()


if __name__ == "__main__":
    main(sys.argv[1:])

