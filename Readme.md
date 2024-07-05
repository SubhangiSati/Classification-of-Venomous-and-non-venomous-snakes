# Snake Classification (venomous/ non-venomous) using Transfer Learning with MobileNetV2

## Overview

This code implements a snake venom classification model using transfer learning with the MobileNetV2 architecture. The model is trained on a dataset of snake images categorized into venomous and non-venomous classes. It evaluates the model's performance, predicts the class for a sample image, and provides a confusion matrix and classification report.

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

Ensure you have the required dependencies installed using:

```bash
pip install tensorflow matplotlib seaborn scikit-learn
```

## Usage

1. Download the snake venom dataset and organize it into train and test directories.

2. Update the `train_dir` and `test_dir` variables with the correct paths to your train and test datasets.

3. Optionally, set the `image_path` variable to the path of a specific image for prediction.

4. Run the script to train the model, evaluate its performance, and make predictions.

```bash
python snake_venom_classification.py
```

## Code Structure

- **Dataset Loading:**
  - The dataset is loaded and preprocessed using image augmentation for training.

- **Model Creation:**
  - MobileNetV2 is employed as the base model with additional custom top layers for classification.
  - The model is compiled with categorical cross-entropy loss and the Adam optimizer.

- **Training:**
  - The model is trained for a specified number of epochs.

- **Evaluation:**
  - The model's accuracy is evaluated on the test set, and an accuracy graph is plotted.

- **Prediction:**
  - A sample image is loaded, preprocessed, and the model predicts its class.

- **Confusion Matrix:**
  - A confusion matrix and classification report are generated for evaluating model performance.

## Hyperparameters

- `img_width` and `img_height`: Input image dimensions (224x224).
- `batch_size`: Batch size for training and testing (32).
- `num_classes`: Number of snake venom classes (2 - venomous and non-venomous).
- `epochs`: Number of training epochs (10).

## Customization

- Adjust the hyperparameters to suit your specific dataset and computing resources.
- Modify the model architecture, learning rate, or image augmentation parameters as needed.

## License

This code is licensed under the [MIT License](LICENSE).

