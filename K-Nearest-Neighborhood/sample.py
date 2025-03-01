import numpy as np


def sample(type: str):
    with open(f"../dataset/{type}.npy", "rb") as f:
        data = np.arange(0)
        for i in range(10):
            value = np.load(f)
            rand = np.random.choice(len(value), 100)
            data = np.append(data, value[rand])
        with open(f"./dataset/{type}.npy", "wb") as f:
            np.save(f, data.reshape(len(data) // 785, 785))


