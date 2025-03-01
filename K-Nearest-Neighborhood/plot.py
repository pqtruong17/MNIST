import numpy as np
from k import k_neighbor
import matplotlib.pyplot as plt


def confusion_matrix(k: int):
    arr = k_neighbor(k).reshape(10, 10) / k
    labels = [f"{i}" for i in range(np.shape(arr)[0])]
    fig, ax = plt.subplots(figsize=(10, 10))  # type:ignore
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
    fig.savefig(f"./graph/k={k}.png", bbox_inches="tight", pad_inches=1)
    plt.show()


confusion_matrix(2)
