# MNIST Digit Classification with CNNs

## Project Overview

This project implements and compares three Convolutional Neural Network (CNN) architectures—LeNet-5, CNN0, and CNN1—for digit classification using the MNIST dataset. The goal is to train and evaluate these models to recognize handwritten digits (0-9) with high accuracy.

## Features
*  Dataset: Utilizes the MNIST dataset, consisting of 60,000 training images and 10,000 test images of handwritten digits.
*  Models:
   * LeNet-5: A classic CNN architecture with two convolutional layers, two pooling layers, and fully connected layers.
   * CNN0: A custom CNN with three convolutional layers, three pooling layers, and increased filter sizes.
   * CNN1: A CNN with two convolutional layers, one pooling layer, dropout for regularization, and a sigmoid activation in the dense layer.    
*  Preprocessing: Includes reshaping, normalizing pixel values, and one-hot encoding labels.
*  Training: Models are trained with configurable epochs and batch sizes, using SGD or Adam optimizers.
*  Evaluation: Visualizes training/validation accuracy and loss, and displays sample predictions.
-------------------------------------------------------------------------------------------------
## Results
* LeNet-5: Achieves ~98-99% test accuracy with 50 epochs.
* CNN0: Achieves ~99% test accuracy with 30 epochs, benefiting from deeper architecture.
* CNN1: Achieves ~98% test accuracy, with dropout improving generalization.

## Notes
* The script includes visualizations for data exploration and model performance.
* Adjust epochs and batch_size in train_model for experimentation.
* CNN0 uses the Adam optimizer, while LeNet-5 and CNN1 use SGD with a learning rate of 0.01.
-------------------------------------------------------------------------------------------------
### File Structure
* CNN.py: Main script containing data loading, preprocessing, model definitions, training, and evaluation functions.
   * Functions:
     * preprocess_data: Reshapes and normalizes data, applies one-hot encoding.
     * train_model: Trains and evaluates models, plots accuracy/loss.
     * summary_history: Visualizes training/validation metrics.
     * example: Displays sample predictions.
     * LeNet, CNN0, CNN1: Model architecture definitions.
-------------------------------------------------------------------------------------------------
### License

MIT License

### Requirements
* Python 3.x
* Keras
* TensorFlow
* NumPy
* Matplotlib
