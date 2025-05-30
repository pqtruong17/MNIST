import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def heatMap(arr, k: int):
    if not os.path.exists("./graphs"):
        os.mkdir("./graphs")
    labels = [f"{i}" for i in range(np.shape(arr)[0])]
    fig, ax = plt.subplots(figsize=(28, 28))  # type:ignore
    ax.imshow(arr, cmap="viridis")
    for i in range(np.shape(arr)[0]):
        for j in range(np.shape(arr)[1]):
            ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", color="white")
    ax.set_xticks(np.arange(np.shape(arr)[1]))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(np.shape(arr)[0]))
    ax.set_yticklabels(labels)
    ax.xaxis.set_ticks_position("top")
    plt.tight_layout()
    fig.savefig(f"./graphs/heatMapOf{k}.png", bbox_inches="tight", pad_inches=1)


def main(args):
    for arg in args:
        if arg == "graphs":
            with open("./weights/weights.npy", "rb") as f:
                for i in range(10):
                    arr = np.load(f)[1:785].reshape(28, 28)
                    heatMap(arr, i)


if __name__ == "__main__":
    main(sys.argv[1:])
