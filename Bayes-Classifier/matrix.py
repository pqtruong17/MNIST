import numpy as np, matplotlib.pyplot as plt, sys, os


def matrix(name: str):
    with open(f"{name}.npy", "rb") as fp:
        arr = np.load(fp)
    labels = [f"{i}" for i in range(np.shape(arr)[0])]
    fig, ax = plt.subplots(figsize=(10, 10))  # type:ignore
    ax.imshow(arr, cmap="viridis")
    for i in range(np.shape(arr)[0]):
        for j in range(np.shape(arr)[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color="white")
    ax.set_xticks(np.arange(np.shape(arr)[1]))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(np.shape(arr)[0]))
    ax.set_yticklabels(labels)
    ax.xaxis.set_ticks_position("top")
    plt.tight_layout()
    fig.savefig(f"{name}.png", bbox_inches="tight", pad_inches=1)
    plt.show()


def main(args):
    matrix("matrix")

if __name__ == "__main__":
    main(sys.argv[1:])
