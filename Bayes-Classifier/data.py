import numpy as np
import os


def sample(type: str, size: int):
    str = ""
    try:
        for i in range(10):
            with open(f"../dataset/{type}{i}.txt", "rt") as file:
                lines = file.readlines()
                for j in range(size):
                    index = np.random.randint(len(lines))
                    str += lines[index]
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error has occurred: {e}")
    return str


def make(type: str):
    txt = sample(f"{type}", 100)
    if os.path.exists(f"./dataset/{type}.txt"):
        f = open(f"./dataset/{type}.txt", "w")
        f.write(txt)
    else:
        f = open(f"./dataset/{type}.txt", "x")
        f = open(f"./dataset/{type}.txt", "w")
        f.write(txt)
    f.close()


make("test")
make("train")
