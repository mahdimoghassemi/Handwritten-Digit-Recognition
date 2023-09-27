# Handwritten-Digit-Recognition
# Image Classification

## Overview

This repository contains code for a deep learning model designed for image classification tasks. The model includes a custom softmax function and is implemented using TensorFlow and Keras. The project provides a comprehensive demonstration of creating, training, and evaluating a neural network for image recognition.

## Table of Contents

- [Neural Network Architecture](#neural-network-architecture)
- [Softmax Function](#softmax-function)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Results and Visualizations](#results-and-visualizations)
- [Dependencies](#dependencies)

## Neural Network Architecture

The neural network architecture is designed for image classification tasks. It consists of the following layers:

- Input Layer: 400 neurons (assuming 20x20 pixel images)
- Hidden Layer 1: 25 neurons with ReLU activation
- Hidden Layer 2: 15 neurons with ReLU activation
- Output Layer: 10 neurons with linear activation

The model is compiled using the sparse categorical cross-entropy loss and the Adam optimizer.

## Softmax Function

The repository includes a custom softmax function (`my_softmax`) implemented in both NumPy and TensorFlow. This function converts a vector of values to a probability distribution and is used in the output layer of the neural network.

## Getting Started

### Prerequisites

Before running the code, ensure you have the following dependencies installed:

- NumPy
- TensorFlow
- Matplotlib

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/deep-learning-image-classification.git
cd deep-learning-image-classification
```

## Results and Visualizations

The model training results, including accuracy and loss curves, can be found in the `results` directory. Additionally, random images from the dataset are visualized alongside the model's predictions to provide insights into its performance.

## Dependencies

- NumPy
- TensorFlow
- Matplotlib

Install dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

