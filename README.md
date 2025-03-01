# EMNIST Handwritten Letters Classification

This repository contains a deep learning model built using TensorFlow and Keras to classify handwritten letters from the EMNIST dataset. The dataset is preprocessed, augmented, and used to train a fully connected neural network.

## Dataset

The model uses the EMNIST dataset, specifically the 'letters' split, which contains images of handwritten letters labeled from A to Z.

## Features

- Uses PyTorch's `torchvision` to download and preprocess the dataset.
- Converts images to NumPy arrays for compatibility with TensorFlow.
- Applies data augmentation using `ImageDataGenerator`.
- Implements a fully connected neural network with Batch Normalization and Dropout layers.
- Trains the model using categorical cross-entropy loss and Adam optimizer.
- Plots training history and visualizes predictions.

## Requirements

To run this project, you need the following Python libraries:

```bash
pip install numpy matplotlib tensorflow torchvision torch
```



## Model Architecture

The model consists of:

- Flatten layer for input preprocessing
- Dense layers with ReLU activation
- Batch Normalization for stable training
- Dropout layers to prevent overfitting
- Softmax activation for final classification

## Training

The model is trained for 20 epochs with a batch size of 64. It uses the Adam optimizer with an initial learning rate of `1e-3`.

## Evaluation

After training, the model is evaluated on the test dataset. The script prints the test accuracy and visualizes the predictions on sample images.

## Results

- Training and validation accuracy plots are displayed.
- Sample predictions are visualized with true and predicted labels.
- finally achieved 90% of Accuracy

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License

This project is licensed under the MIT License.

