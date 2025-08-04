import numpy as np
import requests
import gzip
import os


def download_gzip(url: str, cache_path: str = ".cache") -> bytes:
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    filename = os.path.basename(url)
    filepath = os.path.join(cache_path, filename)

    if filename in os.listdir(cache_path):
        with open(filepath, "rb") as file:
            return gzip.decompress(file.read())

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}")

    data = response.content
    with open(filepath, "wb") as file:
        file.write(data)

    return gzip.decompress(data)


def download_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Downloads the MNIST dataset from Apache SystemDS,
    decompresses the files, and loads them into NumPy arrays.

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
               A tuple of NumPy arrays for training images, training labels,
               test images, and test labels.
    """
    base_url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
    train_images_gz = download_gzip(base_url + 'train-images-idx3-ubyte.gz')
    train_labels_gz = download_gzip(base_url + 'train-labels-idx1-ubyte.gz')
    test_images_gz = download_gzip(base_url + 't10k-images-idx3-ubyte.gz')
    test_labels_gz = download_gzip(base_url + 't10k-labels-idx1-ubyte.gz')

    # Parse image files (idx3-ubyte format)
    # The header is 16 bytes: magic number (4), number of images (4),
    # rows (4), columns (4)
    x_train = np.frombuffer(
        train_images_gz,
        dtype=np.uint8,
        offset=16
    ).reshape(-1, 28, 28)
    x_test = np.frombuffer(
        test_images_gz,
        dtype=np.uint8,
        offset=16
    ).reshape(-1, 28, 28)

    # Parse label files (idx1-ubyte format)
    # The header is 8 bytes: magic number (4), number of items (4)
    y_train = np.frombuffer(train_labels_gz, dtype=np.uint8, offset=8)
    y_test = np.frombuffer(test_labels_gz, dtype=np.uint8, offset=8)
    return x_train, y_train, x_test, y_test
