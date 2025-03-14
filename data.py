import numpy as np
import os


def txtToArr(type: str):
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")
    with open(f"./dataset/{type}.npy", "wb") as f:
        for i in range(10):
            np.save(
                f,
                np.genfromtxt(f"./olddata/{type}{i}.txt", dtype="int32", delimiter=" "),
            )

# txtToArr("train")
# txtToArr("test")
