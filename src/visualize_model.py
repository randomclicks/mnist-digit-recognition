import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import os

def load_data():
    """Load and preprocess MNIST test data."""
    print("Loading MNIST test data...")
    (_, _), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN input
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    return x_test, y_test

def load_saved_model(model_path='models/mnist_model.h5'):
    """Load the trained model."""
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def visualize_predictions(model, x_test, y_test, num_samples=10):
    """Visualize model predictions on random test samples."""
    print("\nGenerating prediction visualizations...")
    
    # Create visualizations directory if it doesn't exist
    vis_dir = 'visualizations/samples'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Get random samples
    indices = np.random.randint(0, x_test.shape[0], num_samples)
    
    # Create figure
    plt.figure(figsize=(15, 6))
    
    for i, idx in enumerate(indices):
        # Get sample and prediction
        img = x_test[idx]
        true_label = y_test[idx]
        
        # Make prediction
        pred = model.predict(img.reshape(1, 28, 28, 1), verbose=0)
        pred_label = np.argmax(pred)
        confidence = np.max(pred) * 100
        
        # Plot
        plt.subplot(2, 5, i + 1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                 color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'sample_predictions.png'))
    plt.close()
    
    print("Sample predictions visualization saved as 'visualizations/samples/sample_predictions.png'")

def visualize_confusion_matrix(model, x_test, y_test):
    """Generate and plot confusion matrix."""
    print("\nGenerating confusion matrix...")
    
    # Create visualizations directory if it doesn't exist
    vis_dir = 'visualizations/analysis'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Get predictions
    predictions = model.predict(x_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Calculate confusion matrix
    conf_matrix = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_test, pred_classes):
        conf_matrix[true, pred] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar()
    
    # Add labels and title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Add text annotations
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(conf_matrix[i, j]),
                    ha='center', va='center')
    
    plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'))
    plt.close()
    
    print("Confusion matrix saved as 'visualizations/analysis/confusion_matrix.png'")

def visualize_confidence_distribution(model, x_test, y_test):
    """Visualize the distribution of prediction confidences."""
    print("\nGenerating confidence distribution plot...")
    
    # Create visualizations directory if it doesn't exist
    vis_dir = 'visualizations/analysis'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Get predictions and confidences
    predictions = model.predict(x_test, verbose=0)
    confidences = np.max(predictions, axis=1) * 100
    correct = (np.argmax(predictions, axis=1) == y_test)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(confidences[correct], bins=50, alpha=0.5, label='Correct Predictions',
             color='green', density=True)
    plt.hist(confidences[~correct], bins=50, alpha=0.5, label='Incorrect Predictions',
             color='red', density=True)
    
    # Add labels and title
    plt.xlabel('Confidence (%)')
    plt.ylabel('Density')
    plt.title('Distribution of Model Confidence')
    plt.legend()
    
    plt.savefig(os.path.join(vis_dir, 'confidence_distribution.png'))
    plt.close()
    
    print("Confidence distribution plot saved as 'visualizations/analysis/confidence_distribution.png'")

def analyze_errors(model, x_test, y_test):
    """Analyze and visualize the most confident errors."""
    print("\nAnalyzing prediction errors...")
    
    # Create visualizations directory if it doesn't exist
    vis_dir = 'visualizations/errors'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Get predictions
    predictions = model.predict(x_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1) * 100
    
    # Find errors
    errors = pred_classes != y_test
    error_indices = np.where(errors)[0]
    
    if len(error_indices) == 0:
        print("No errors found in test set!")
        return
    
    # Sort errors by confidence
    error_confidences = confidences[error_indices]
    sorted_indices = error_indices[np.argsort(error_confidences)][::-1]
    
    # Plot top 10 most confident errors
    num_errors = min(10, len(sorted_indices))
    plt.figure(figsize=(15, 6))
    
    for i in range(num_errors):
        idx = sorted_indices[i]
        img = x_test[idx]
        true_label = y_test[idx]
        pred_label = pred_classes[idx]
        confidence = confidences[idx]
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                 color='red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'confident_errors.png'))
    plt.close()
    
    print("Most confident errors visualization saved as 'visualizations/errors/confident_errors.png'")

def main():
    """Main function to generate model visualizations."""
    print("MNIST Model Visualization")
    print("========================")
    
    # Load test data
    x_test, y_test = load_data()
    
    # Load model
    model = load_saved_model()
    if model is None:
        return
    
    # Generate visualizations
    visualize_predictions(model, x_test, y_test)
    visualize_confusion_matrix(model, x_test, y_test)
    visualize_confidence_distribution(model, x_test, y_test)
    analyze_errors(model, x_test, y_test)
    
    print("\nVisualization complete! Check the 'visualizations' directory for all plots.")

if __name__ == "__main__":
    main()