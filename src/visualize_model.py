import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape the images to add the channel dimension (MNIST is grayscale)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

def visualize_confusion_matrix(model):
    """Visualize the confusion matrix of model predictions."""
    # Get predictions
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(test_labels, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('visualizations/confusion_matrix.png')
    plt.show()
    
    # Calculate and print classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_classes))

def visualize_feature_maps(model, image_idx=0):
    """Visualize feature maps from different convolutional layers."""
    # Create a model that outputs feature maps from each convolutional layer
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name.lower()]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get feature maps for a single image
    img = test_images[image_idx:image_idx+1]
    activations = activation_model.predict(img)
    
    # Display the original image
    plt.figure(figsize=(8, 8))
    plt.imshow(test_images[image_idx].reshape(28, 28), cmap='gray')
    plt.title(f'Original Image (Label: {test_labels[image_idx]})')
    plt.axis('off')
    plt.savefig('visualizations/original_image.png')
    plt.show()
    
    # Display feature maps from each layer
    for i, layer_activation in enumerate(activations):
        # Get the number of features in the feature map
        n_features = layer_activation.shape[-1]
        
        # Determine grid size
        size = int(np.ceil(np.sqrt(n_features)))
        
        # Create figure with subplots
        fig, axs = plt.subplots(size, size, figsize=(12, 12))
        fig.suptitle(f'Feature Maps from Convolutional Layer {i+1}', fontsize=16)
        
        # Plot each feature map
        for j in range(n_features):
            row, col = j // size, j % size
            ax = axs[row, col]
            ax.imshow(layer_activation[0, :, :, j], cmap='viridis')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide empty subplots
        for j in range(n_features, size*size):
            row, col = j // size, j % size
            axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(f'visualizations/feature_maps_layer{i+1}.png')
        plt.show()

def visualize_embedding(model, samples=5000):
    """Visualize digit embeddings using t-SNE."""
    # Create a model that outputs the features before classification
    feature_model = tf.keras.models.Model(inputs=model.input, 
                                          outputs=model.layers[-2].output)
    
    # Get features for a subset of test images
    features = feature_model.predict(test_images[:samples])
    labels = test_labels[:samples]
    
    # Use t-SNE to reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot t-SNE visualization
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(10):
        plt.scatter(features_2d[labels == i, 0], features_2d[labels == i, 1],
                    color=colors[i], label=f"Digit {i}", alpha=0.7)
    
    plt.title('t-SNE Visualization of Digit Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/tsne_embedding.png')
    plt.show()

def visualize_misclassified(model, num_samples=10):
    """Visualize some misclassified digits."""
    # Get predictions
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Find misclassified examples
    misclassified_indices = np.where(predicted_classes != test_labels)[0]
    
    if len(misclassified_indices) > 0:
        # Select random misclassified examples
        selected_indices = np.random.choice(misclassified_indices, 
                                           min(num_samples, len(misclassified_indices)), 
                                           replace=False)
        
        # Plot misclassified examples
        plt.figure(figsize=(15, 3))
        for i, idx in enumerate(selected_indices):
            plt.subplot(1, num_samples, i+1)
            plt.imshow(test_images[idx].reshape(28, 28), cmap='gray')
            plt.title(f"True: {test_labels[idx]}\nPred: {predicted_classes[idx]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/misclassified_examples.png')
        plt.show()
    else:
        print("No misclassified examples found!")

if __name__ == "__main__":
    # Load trained model
    try:
        model = load_model('models/mnist_model.h5')
        print("Model loaded successfully!")
        
        # Visualization menu
        while True:
            print("\nVisualization Options:")
            print("1. Confusion Matrix")
            print("2. Feature Maps")
            print("3. t-SNE Embedding")
            print("4. Misclassified Examples")
            print("5. All Visualizations")
            print("6. Exit")
            
            choice = input("Enter your choice (1-6): ")
            
            if choice == '1':
                visualize_confusion_matrix(model)
            elif choice == '2':
                image_idx = int(input("Enter image index (0-9999): ") or "0")
                visualize_feature_maps(model, image_idx)
            elif choice == '3':
                samples = int(input("Enter number of samples (default: 5000): ") or "5000")
                visualize_embedding(model, samples)
            elif choice == '4':
                num_samples = int(input("Enter number of examples to show (default: 10): ") or "10")
                visualize_misclassified(model, num_samples)
            elif choice == '5':
                print("Generating all visualizations...")
                visualize_confusion_matrix(model)
                visualize_feature_maps(model)
                visualize_embedding(model)
                visualize_misclassified(model)
            elif choice == '6':
                print("Exiting...")
                break
            else:
                print("Invalid choice!")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to train the model first by running mnist_digit_recognition.py")