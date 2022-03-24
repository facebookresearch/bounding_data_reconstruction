# Bounding Training Data Reconstruction in Private (Deep) Learning

This repository contains code for reproducing results in the paper:
- Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten. **[Bounding Training Data Reconstruction in Private (Deep) Learning](https://arxiv.org/abs/2201.12383)**.

## Setup

Dependencies: [hydra](https://github.com/facebookresearch/hydra), [numpy](https://numpy.org/), [pytorch](https://pytorch.org/), [fisher_information_loss](https://github.com/facebookresearch/fisher_information_loss)

For private SGD experiments: [jax](https://github.com/google/jax), [tensorflow-privacy](https://github.com/tensorflow/privacy)

After installing dependencies, download the [fisher_information_loss](https://github.com/facebookresearch/fisher_information_loss) submodule:
```
git submodule update --init
```

## Experiments

### MNIST Logistic Regression

Trains a logistic regression model for MNIST 0 vs. 1 classification and compute RDP and FIL privacy accounting:
```
python mnist_logistic_regression.py --lam 1e-2 --sigma 1e-2
```

Runs the [Balle et al.](https://arxiv.org/abs/2201.04845) GLM attack on the logistic regression model:
```
python mnist_logistic_reconstruction.py --lam 1e-2 --sigma 1e-5
```

### Private SGD Training

Trains a private model on MNIST/CIFAR-10 with RDP and FIL privacy accounting:
```
python train_classifier.py --config-name [mnist.yaml/cifar.yaml]
```
Check `configs` directory for Hydra configs, and see appendix in our paper for the full grid of hyperparameter values.

## Code Acknowledgements

The majority of Bounding Data Reconstruction is licensed under CC-BY-NC, however portions of the project are available under separate terms: [hydra](https://github.com/facebookresearch/hydra) is licensed under the MIT license; and [jax](https://github.com/google/jax) and [tensorflow-privacy](https://github.com/tensorflow/privacy) are licensed under the Apache 2.0 license.
