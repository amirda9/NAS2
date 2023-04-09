# Deep Learning Course Project
## Adversarial NAS

Welcome to my Deep Learning Course Project! In this project, I try to reproduce the AdversarialNAS to find the the super GAN for cifar10 and stl10 datasets. And I also I try to find the super GAN for imagenette dataset to see if it works well or not.

## Problem Statement

My problem statement is to build a superior GAN in order to generate images that are more realistic than the original images. The original images are from the CIFAR10 and STL10 datasets. The GAN is trained on the CIFAR10 and STL10 datasets. The GAN is then used to generate images that are more realistic than the original images.

## Datasets

This repository uses the following datasets:

- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [STL-10](https://cs.stanford.edu/~acoates/stl10/)
- [Imagenette](https://github.com/fastai/imagenette)

## Model Architecture

How to run:

1. Clone the repository
2. Install the requirements
3. Run the following command to train the GAN on CIFAR10 dataset:

```bash
python3 search.py --mode search --data cifar10 
```

4. Run the following command to train the GAN on STL10 dataset:

```bash
python3 search.py --mode search --data stl10 
```

5. Run the following command to train the GAN on Imagenette dataset:

```bash
python3 search.py --mode search --data imagenette 
```


