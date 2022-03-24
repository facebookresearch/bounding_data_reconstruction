#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import array
import gzip
import logging
import os
from os import path
import struct
import math
import urllib.request
from torchvision import datasets as torch_datasets
from torchvision import transforms

import numpy as np
import numpy.random as npr
from sklearn.decomposition import PCA


_DATA_FOLDER = "data/"


def _download(url, data_folder, filename):
    """
    Download a URL to a file in the temporary data directory, if it does not
    already exist.
    """
    if not path.exists(data_folder):
        os.makedirs(data_folder)
    out_file = path.join(data_folder, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        logging.info(f"Downloaded {url} to {data_folder}")


def _partial_flatten(x):
    """
    Flatten all but the first dimension of an ndarray.
    """
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
    """
    Create a one-hot encoding of x of size k.
    """
    return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw(dataset):
    """
    Download and parse the raw MNIST dataset.
    """

    if dataset == "mnist":
        # mirror of http://yann.lecun.com/exdb/mnist/:
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    elif dataset == "fmnist":
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    elif dataset == "kmnist":
        base_url = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/"
    else:
        raise RuntimeError("Unknown dataset: " + dataset)
    data_folder = path.join(_DATA_FOLDER, dataset)

    def parse_labels(filename):
        """
        Parses labels in MNIST raw label file.
        """
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        """
        Parses images in MNIST raw label file.
        """
        with gzip.open(filename, "rb") as fh:
            _, num_DATA_FOLDER, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_DATA_FOLDER, rows, cols
            )

    # download all MNIST files:
    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        _download(base_url + filename, data_folder, filename)

    # parse all images and labels:
    train_images = parse_images(path.join(data_folder, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(data_folder, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(data_folder, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(data_folder, "t10k-labels-idx1-ubyte.gz"))
    return train_images, train_labels, test_images, test_labels


def preprocess_data(train_images, train_labels, test_images, test_labels,
                    binary, permute_train, normalize, pca_dims):
    if binary:
        num_labels = 2
        train_mask = np.logical_or(train_labels == 0, train_labels == 1)
        test_mask = np.logical_or(test_labels == 0, test_labels == 1)
        train_images, train_labels = train_images[train_mask], train_labels[train_mask]
        test_images, test_labels = test_images[test_mask], test_labels[test_mask]
    else:
        num_labels = np.max(test_labels) + 1
    train_labels = _one_hot(train_labels, num_labels)
    test_labels = _one_hot(test_labels, num_labels)
    
    if pca_dims > 0:
        pca = PCA(n_components=pca_dims, svd_solver='full')
        pca.fit(train_images)
        train_images = pca.transform(train_images)
        test_images = pca.transform(test_images)
        
    if normalize:
        train_images /= np.linalg.norm(train_images, 2, 1)[:, None]
        test_images /= np.linalg.norm(test_images, 2, 1)[:, None]

    # permute training data:
    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]
    return train_images, train_labels, test_images, test_labels


def mnist(dataset="mnist", binary=False, permute_train=False, normalize=False, pca_dims=0):
    """
    Download, parse and process MNIST data to unit scale and one-hot labels.
    """

    # obtain raw MNIST data:
    train_images, train_labels, test_images, test_labels = mnist_raw(dataset)

    # flatten and normalize images, create one-hot labels:
    train_images = _partial_flatten(train_images) / np.float32(255.0)
    test_images = _partial_flatten(test_images) / np.float32(255.0)
    
    return preprocess_data(train_images, train_labels, test_images, test_labels,
                           binary, permute_train, normalize, pca_dims)


def cifar(dataset="cifar10", binary=False, permute_train=False, normalize=False, pca_dims=0):
    
    data_folder = path.join(_DATA_FOLDER, dataset)
    normalizer = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transforms = transforms.Compose([transforms.ToTensor(), normalizer])
    if dataset == "cifar10":
        train_set = torch_datasets.CIFAR10(root=data_folder, train=True, transform=train_transforms, download=True)
        test_set = torch_datasets.CIFAR10(root=data_folder, train=False, transform=train_transforms, download=True)
    elif dataset == "cifar100":
        train_set = torch_datasets.CIFAR100(root=data_folder, train=True, transform=train_transforms, download=True)
        test_set = torch_datasets.CIFAR100(root=data_folder, train=False, transform=train_transforms, download=True)
    
    train_images = []
    train_labels = []
    for (x, y) in train_set:
        train_images.append(np.rollaxis(x.numpy(), 0, 3).flatten())
        train_labels.append(y)
    train_images = np.stack(train_images)
    train_labels = np.array(train_labels)
    
    test_images = []
    test_labels = []
    for (x, y) in test_set:
        test_images.append(np.rollaxis(x.numpy(), 0, 3).flatten())
        test_labels.append(y)
    test_images = np.stack(test_images)
    test_labels = np.array(test_labels)
    
    return preprocess_data(train_images, train_labels, test_images, test_labels,
                           binary, permute_train, normalize, pca_dims)


def get_datastream(images, labels, batch_size, permutation=False, last_batch=True):
    """
    Returns a data stream of `images` and corresponding `labels` in batches of
    size `batch_size`. Also returns the number of batches per epoch, `num_batches`.

    To loop through the whole dataset in permuted order, set `permutation` to `True`.
    To not return the last batch, set `last_batch` to `False`.
    """

    # compute number of batches to return:
    num_images = images.shape[0]

    def permutation_datastream():
        """
        Data stream iterator that returns randomly permuted images until eternity.
        """
        while True:
            perm = npr.permutation(num_images)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield images[batch_idx], labels[batch_idx], batch_idx
                
    def random_sampler_datastream():
        """
        Data stream iterator that returns a uniformly random batch of images until eternity.
        """
        while True:
            batch_idx = npr.permutation(num_images)[:batch_size]
            yield images[batch_idx], labels[batch_idx], batch_idx
    
    # return iterator factory:
    if permutation:
        num_batches = int((math.ceil if last_batch else math.floor)(float(num_images) / float(batch_size)))
        return random_sampler_datastream, num_batches
    else:
        num_complete_batches, leftover = divmod(num_images, batch_size)
        num_batches = num_complete_batches + (last_batch and bool(leftover))
        return permutation_datastream, num_batches
