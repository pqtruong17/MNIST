import numpy as np


def txtToArr(type: str):
    with open(f"./dataset/{type}.npy", "wb") as f:
        for i in range(10):
            np.save(
                f,
                np.genfromtxt(f"./olddata/{type}{i}.txt", dtype="int32", delimiter=" "),
            )

