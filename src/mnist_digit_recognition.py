import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_preprocess_data():
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN input
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def create_model():
    """Create and compile the CNN model."""
    print("Creating CNN model...")
    model = models.Sequential([
        # First Convolutional Layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=10):
    """Train the model and return training history."""
    print("\nTraining model...")
    history = model.fit(x_train, y_train,
                       batch_size=128,
                       epochs=epochs,
                       validation_data=(x_test, y_test),
                       verbose=1)
    return history

def evaluate_model(model, x_test, y_test):
    """Evaluate model performance."""
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy*100:.1f}%")
    print(f"Test loss: {test_loss:.4f}")
    return test_accuracy, test_loss

def plot_training_history(history):
    """Plot training history."""
    print("\nPlotting training history...")
    
    # Create directory for visualizations if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig('visualizations/training_history.png')
    plt.close()
    
    print("Training history plot saved as 'visualizations/training_history.png'")

def save_model(model):
    """Save the trained model."""
    print("\nSaving model...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model.save('models/mnist_model.h5')
    print("Model saved as 'models/mnist_model.h5'")

def main():
    """Main function to train and evaluate the MNIST digit recognition model."""
    print("MNIST Digit Recognition - Model Training")
    print("=======================================")
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create model
    model = create_model()
    model.summary()
    
    # Train model
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Evaluate model
    evaluate_model(model, x_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    save_model(model)
    
    print("\nTraining complete! You can now use the model with draw_and_recognize.py")

if __name__ == "__main__":
    main()