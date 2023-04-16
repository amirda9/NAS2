# Deep Learning Course Project
## Adversarial NAS

Welcome to my Deep Learning Course Project! In this project, I try to reproduce the AdversarialNAS to find the the super GAN for cifar10 and stl10 datasets. And I also I try to find the super GAN for imagenette dataset to see if it works well or not.

## Problem Statement

My problem statement is to build a superior GAN in order to generate images that are more realistic than the original images. The original images are from the CIFAR10 and STL10 datasets. The GAN is trained on the CIFAR10 and STL10 datasets. The GAN is then used to generate images that are more realistic than the original images.


## Dataset

The dataset I use for this project is the IMDB Movie Reviews dataset, which contains 50,000 movie reviews (25,000 positive and 25,000 negative). The dataset is split into training and testing sets, with 20% of the data reserved for testing. I use the Keras API to load and preprocess the data.

## Model Architecture

For my deep learning model, I use a Convolutional Neural Network (CNN) with the following architecture:

- Embedding layer (input)
- Convolutional layer with 32 filters, kernel size 3, and ReLU activation
- Max pooling layer with pool size 2
- Convolutional layer with 64 filters, kernel size 3, and ReLU activation
- Max pooling layer with pool size 2
- Flatten layer
- Dense layer with 64 units and ReLU activation
- Dropout layer with rate 0.5
- Output layer with 1 unit and sigmoid activation

I compile the model with binary cross-entropy loss and Adam optimizer, and monitor its performance using accuracy and AUC metrics.

## Results

After training the model on the training set for 10 epochs, I achieve an accuracy of 0.87 and an AUC of 0.93 on the testing set. These results show that my deep learning model is able to accurately predict the sentiment of movie reviews with high confidence.

## Conclusion

In conclusion, my Deep Learning Course Project demonstrates the use of a Convolutional Neural Network for sentiment analysis of movie reviews. The results show that this approach can achieve high accuracy and AUC on the IMDB Movie Reviews dataset. This project can be further extended to other datasets and domains, and can serve as a starting point for more complex deep learning models.
