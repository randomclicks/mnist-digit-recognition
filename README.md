# MNIST Digit Recognition Project

A comprehensive implementation of a handwritten digit recognition system using the MNIST dataset and deep learning.

## Overview

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) from the MNIST dataset. The implementation includes:

- A neural network model training pipeline
- Testing tools for evaluating on new images
- Interactive applications for real-time recognition
- Visualization tools for understanding the model's behavior

## Project Structure

```
mnist-digit-recognition/
├── src/                    # Source code
│   ├── mnist_digit_recognition.py  # Main training script
│   ├── test_custom_digits.py       # Custom image testing
│   ├── visualize_model.py          # Model visualization tools
│   └── draw_and_recognize.py       # Interactive drawing app
├── models/                 # Trained model files
│   └── mnist_model.h5      # Trained CNN model
├── data/                  # Dataset and data-related files
├── visualizations/        # Generated visualizations
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── feature_maps_*.png
│   └── tsne_embedding.png
├── docs/                  # Documentation
├── saved_drawings/        # User-created drawings
└── requirements.txt       # Project dependencies
```

## Model Performance

The implemented CNN model achieves:
- Training accuracy: ~99%
- Validation accuracy: ~98-99%
- Test accuracy: ~99%

Key features of the model:
- 3 convolutional layers with ReLU activation
- Max pooling for dimensionality reduction
- Dropout layer for regularization
- Dense layers for final classification

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/randomclicks/mnist-digit-recognition.git
   cd mnist-digit-recognition
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Train the CNN model on the MNIST dataset:
```bash
python src/mnist_digit_recognition.py
```

This will:
- Download and prepare the MNIST dataset
- Train the model for 10 epochs
- Save the trained model as `models/mnist_model.h5`
- Generate performance visualizations in `visualizations/`

### Interactive Drawing Application

Test the model by drawing digits yourself:
```bash
python src/draw_and_recognize.py
```

Features:
- Modern UI with light/dark themes
- Real-time digit recognition
- Confidence visualization for all digits
- Grid overlay for drawing guidance
- Save/load drawings
- Adjustable brush size

### Testing Custom Images

Test the model with your own digit images or webcam:
```bash
python src/test_custom_digits.py
```

### Model Visualization

Explore model behavior and performance:
```bash
python src/visualize_model.py
```

Options include:
1. Confusion matrix
2. Feature maps
3. t-SNE embedding visualization
4. Misclassified examples analysis

## Project Components

### 1. Model Architecture
- Input layer: 28x28x1 (grayscale images)
- Conv2D layers: 32, 64, and 64 filters
- MaxPooling2D layers for downsampling
- Dense layers: 64 units with dropout
- Output layer: 10 units (softmax)

### 2. Interactive Drawing App
- PyGame-based interface
- Real-time prediction
- Modern UI elements
- Multiple visualization options

### 3. Visualization Tools
- Training/validation curves
- Confusion matrix
- Feature map visualization
- t-SNE embedding visualization
- Misclassification analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MNIST dataset providers
- TensorFlow and Keras teams
- PyGame community