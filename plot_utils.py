import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pickle as pkl
import patch


def plot_topic(topic_images: np.ndarray or torch.Tensor, path: str, mode: str, topic: int, n: int = 10):
    topic_path = os.path.join(path, mode)
    if not os.path.exists(topic_path):
        os.mkdir(topic_path)
    topic_file = os.path.join(topic_path, f"{mode}_topic_{topic}.png")
    _, axs = plt.subplots(n // 2, 2, figsize=(10, 20))
    for i, (row, ax) in enumerate(zip(topic_images, axs.flatten())):
        if i >= n:
            break
        if len(row.shape) != 3:
            row = row.reshape(28, 28, 1)
        ax.imshow(row)
    plt.savefig(topic_file)


def plot_histograms(topics, possible, possible_topics, labels, path, mode):
    if not os.path.exists(os.path.join(path, mode, "distributions")):
        os.makedirs(os.path.join(path, mode, "distributions"), exist_ok=True)
    combined_entropy = []
    for i, p in enumerate(possible):
        plt.figure()
        plt.title(f"Topic {i}")
        plt.xlabel("Class Label")
        plt.ylabel("Counts")
        plt.hist(labels[p], bins=possible_topics)
        combined_entropy.append(patch.compute_entropy(labels[p], topics))
        plt.savefig(os.path.join(path, mode, "distributions", f"{mode}_distribution_{i}.png"))
        plt.close()
    with open(os.path.join(path, f"{mode}_entropy.pkl"), 'wb+') as f:
        pkl.dump(combined_entropy, f)
    return combined_entropy
