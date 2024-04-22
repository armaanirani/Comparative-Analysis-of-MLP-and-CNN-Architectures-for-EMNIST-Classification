# Neural Network Models for Handwritten Character Recognition

This repository contains the implementation of two neural network architectures, the Multilayer Perceptron (MLP) and Convolutional Neural Network (CNN), designed to solve the task of handwritten character recognition using the EMNIST dataset. The project focuses on exploring various hyperparameters and their impact on model performance, aiming to provide insights into practical applications in image classification.

## Project Overview

### Objectives
- Implement MLP and CNN models to recognize handwritten characters.
- Explore and tune hyperparameters such as activation functions, optimizers, and learning rate schedulers.
- Compare the performance of MLP and CNN models to determine their suitability for image classification tasks.

### Dataset
The project utilizes the EMNIST balanced dataset, an extension of the MNIST dataset of handwritten letters, enhancing diversity and size for robust model training.

## Getting Started

### Prerequisites
- Python 3.12
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, TensorFlow, Keras, Keras-Tuner

### Installation
Clone this repository to your local machine:

git clone https://github.com/armaanirani/Hyperparameter-tuning.git
cd Hyperparameter-tuning

## Models

### Multilayer Perceptron (MLP)
- Configuration: Three hidden layers with varying nodes using ReLU activation, and a softmax output layer.
- Optimizer: Adam

### Convolutional Neural Network (CNN)
- Configuration: Two convolutional layers followed by pooling layers, a fully connected layer, and a softmax output layer.
- Optimizer: Adam

## Hyperparameter Tuning
Hyperparameter tuning was performed using Keras Tuner with techniques such as random search and Bayesian optimization. Key hyperparameters tuned include learning rate, number of layers, activation functions, and dropout rates.

## Results
Detailed evaluation of the models is available in the Jupyter notebook provided. The CNN model demonstrated superior performance in accuracy, precision, recall, and F1-score compared to the MLP model.

## Contributing
Contributions are welcome. Please open an issue first to discuss what you would like to change or add.

## References
Links to the used datasets and libraries are included in the `report.pdf` file.

## Contact
Your Name - armaanirani@gmail.com
