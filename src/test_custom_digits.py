import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

def load_saved_model(model_path='models/mnist_model.h5'):
    """Load the trained model."""
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for prediction."""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28))
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array, img
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def predict_digit(model, img_array):
    """Make prediction for a single image."""
    if model is None or img_array is None:
        return None, None
    
    # Get prediction
    prediction = model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return predicted_digit, confidence

def plot_prediction(img, predicted_digit, confidence, save_path=None):
    """Plot the image with prediction results."""
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}\nConfidence: {confidence:.1f}%')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction plot saved as {save_path}")
    else:
        plt.show()
    plt.close()

def test_saved_drawings(model_path='models/mnist_model.h5', drawings_dir='saved_drawings'):
    """Test the model on saved drawings."""
    print("\nTesting Custom Digits")
    print("====================")
    
    # Load model
    model = load_saved_model(model_path)
    if model is None:
        return
    
    # Create visualizations directory if it doesn't exist
    vis_dir = 'visualizations/predictions'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Get list of saved drawings
    if not os.path.exists(drawings_dir):
        print(f"No drawings found in {drawings_dir}")
        return
    
    drawings = [f for f in os.listdir(drawings_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not drawings:
        print("No image files found in the drawings directory")
        return
    
    print(f"\nFound {len(drawings)} drawings to test")
    results = []
    
    # Process each drawing
    for i, drawing in enumerate(drawings, 1):
        print(f"\nProcessing drawing {i}/{len(drawings)}: {drawing}")
        image_path = os.path.join(drawings_dir, drawing)
        
        # Load and preprocess image
        img_array, img = load_and_preprocess_image(image_path)
        if img_array is None:
            continue
        
        # Make prediction
        predicted_digit, confidence = predict_digit(model, img_array)
        if predicted_digit is None:
            continue
        
        # Store results
        results.append({
            'filename': drawing,
            'predicted': predicted_digit,
            'confidence': confidence
        })
        
        # Plot and save prediction
        save_path = os.path.join(vis_dir, f'prediction_{drawing}')
        plot_prediction(img, predicted_digit, confidence, save_path)
    
    # Print summary
    if results:
        print("\nPrediction Summary:")
        print("------------------")
        for result in results:
            print(f"File: {result['filename']}")
            print(f"Predicted Digit: {result['predicted']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print("------------------")
        
        # Calculate average confidence
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\nAverage confidence across all predictions: {avg_confidence:.1f}%")
        print(f"\nPrediction visualizations saved in {vis_dir}")
    else:
        print("\nNo successful predictions made")

if __name__ == "__main__":
    test_saved_drawings()