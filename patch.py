import numpy as np
import os
import cv2 as cv
from sklearn.cluster import KMeans
from PIL import Image
import random
from multiprocessing import Pool

IMAGE_DIR = "jpg/"

sift = cv.SIFT_create()
files = sorted(os.listdir(IMAGE_DIR))


def load_image(k: int):
    return np.asarray(Image.open(os.path.join(IMAGE_DIR, files[k])).convert("RGB"))


def load_batch_images(start: int = 0, end: int = None):
    if end is None:
        end = len(files)
    return [load_image(k) for k in range(start, end)]


def get_all_patches(image: np.ndarray, k: int) -> [np.ndarray]:
    return [image[i:min(i + k, image.shape[0]), j:min(j + k, image.shape[1]), :] for j in range(0, image.shape[1], k)
            for i in range(0, image.shape[0], k)]


def get_random_patch(image: np.ndarray, k: int, count: int = 1):
    if count == 1:
        i, j = random.randint(0, image.shape[0] - k - 1), random.randint(0, image.shape[1] - k - 1)
        return image[i:i + k, j:j + k, :]
    else:
        assert count > 1
        patches = []
        for _ in range(count):
            i, j = random.randint(0, image.shape[0] - k - 1), random.randint(0, image.shape[1] - k - 1)
            patches.append(image[i:i + k, j:j + k, :])
        return np.stack(patches)


def get_histogram(s: np.ndarray, bins: int or np.asarray):
    return np.histogram(s, bins)


def cluster(vectors: np.ndarray, k: int):
    return KMeans(k).fit(vectors)


def get_sift(image: np.ndarray):
    return sift.detectAndCompute(image, None)


def get_bins(bin_count: int):
    return np.arange(bin_count + 1)


def transform(image, color_quantization, histogram_cluster, color_bin_count, histogram_count, window_size):
    color_bins = get_bins(color_bin_count)
    hist_bins = get_bins(histogram_count)
    histogram_quantized = []
    for p in get_all_patches(image, window_size):
        hist, _ = get_histogram(color_quantization.predict(p.reshape(-1, 3)), color_bins)
        histogram_quantized.append(histogram_cluster.predict(hist.reshape(1, -1)).flatten().item())
    return np.asarray(np.histogram(histogram_quantized, hist_bins)[0])


def get_histograms(images, color_quantization, histogram_cluster, color_bin_count, histogram_count, window_size):
    with Pool(12) as pool:
        return pool.starmap(transform,
                            [(image, color_quantization, histogram_cluster, color_bin_count, histogram_count,
                              window_size) for image in images])


def transform_no_color(image, histogram_cluster, color_bin_count, histogram_count, window_size):
    color_bins = get_bins(color_bin_count)
    hist_bins = get_bins(histogram_count)
    histogram_quantized = []
    for p in get_all_patches(image, window_size):
        hist, _ = get_histogram(p.flatten(), color_bins)
        histogram_quantized.append(histogram_cluster.predict(hist.reshape(1, -1)).flatten().item())
    return np.asarray(np.histogram(histogram_quantized, hist_bins)[0])


def get_histograms_no_color(images, histogram_cluster, color_bin_count, histogram_count, window_size):
    with Pool(10) as pool:
        return pool.starmap(transform_no_color,
                            [(image, histogram_cluster, color_bin_count, histogram_count, window_size) for image in
                             images])


def compute_entropy(p, classes):
    counts = np.bincount(p, minlength=classes)
    freq = counts / np.sum(counts, dtype=np.float)
    return -np.sum(np.log2(freq + 0.00001) * freq)


def get_similar(topics, row, n=5):
    return np.argsort(topics @ row.reshape(-1, 1))[-n:][::-1]
