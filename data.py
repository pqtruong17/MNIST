import numpy as np, os, sys

def sample(type: str):
    with open(f"./dataset/{type}.npy", "wb") as f:
        for i in range(10):
            np.save(f, np.genfromtxt(f"./olddata/{type}{i}.txt", dtype="int32", delimiter=" "),)

def main(args):
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")
        sample("test")
        sample("train")

if __name__ == "__main__":
    main(sys.argv[1:])
