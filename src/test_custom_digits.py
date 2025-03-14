import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
import os

def load_and_preprocess_image(image_path):
    """Load and preprocess an image to match MNIST format."""
    # Read the image
    if isinstance(image_path, str):
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} does not exist")
            return None
            
        # Load from file
        img = Image.open(image_path).convert('L')  # Convert to grayscale
    else:
        # Assume it's already a PIL Image or numpy array
        img = Image.fromarray(image_path) if isinstance(image_path, np.ndarray) else image_path
        img = img.convert('L')  # Convert to grayscale
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)
    
    # Invert if necessary (MNIST has white digits on black background)
    # This step depends on your input images - adjust as needed
    pixel_mean = np.mean(np.array(img))
    if pixel_mean > 128:  # If background is bright
        img = ImageOps.invert(img)
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype('float32') / 255.0
    
    # Reshape to match model input shape
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

def predict_digit(model, image_path):
    """Predict digit from image."""
    # Preprocess image
    processed_img = load_and_preprocess_image(image_path)
    if processed_img is None:
        return None
    
    # Get prediction
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    
    return predicted_class, confidence, processed_img

def draw_prediction(image_path, predicted_digit, confidence, processed_img):
    """Display the original image and the processed image with prediction."""
    plt.figure(figsize=(10, 5))
    
    # Display original image
    plt.subplot(1, 2, 1)
    if isinstance(image_path, str) and os.path.exists(image_path):
        img = plt.imread(image_path)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    else:
        plt.imshow(image_path, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # Display processed image with prediction
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_digit} (Conf: {confidence:.2f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def use_webcam():
    """Use webcam to capture digits and make predictions."""
    model = load_model('models/mnist_model.h5')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Instructions:")
    print("1. Hold a paper with a handwritten digit in front of the camera")
    print("2. Press 'c' to capture and predict")
    print("3. Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Display frame
        cv2.imshow('Digit Recognition', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # If 'c' is pressed, capture image and predict
        if key == ord('c'):
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Add a small ROI in the center for capturing the digit
            h, w = gray.shape
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 3
            roi = gray[center_y-size:center_y+size, center_x-size:center_x+size]
            
            # Predict digit
            digit, confidence, processed_img = predict_digit(model, roi)
            print(f"Predicted digit: {digit} (Confidence: {confidence:.2f}%)")
            
            # Display prediction
            cv2.putText(frame, f"Digit: {digit} ({confidence:.2f}%)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Prediction', frame)
            cv2.waitKey(2000)  # Show for 2 seconds
        
        # If 'q' is pressed, quit
        elif key == ord('q'):
            break
    
    # Release webcam
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load trained model
    model = load_model('models/mnist_model.h5')
    
    # Test mode selection
    print("Select test mode:")
    print("1. Test with a single image file")
    print("2. Use webcam for real-time digit recognition")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == '1':
        # Test with an image file
        image_path = input("Enter path to image file: ")
        
        # Predict digit
        result = predict_digit(model, image_path)
        if result:
            digit, confidence, processed_img = result
            print(f"Predicted digit: {digit}")
            print(f"Confidence: {confidence:.2f}%")
            
            # Draw prediction
            draw_prediction(image_path, digit, confidence, processed_img)
    
    elif choice == '2':
        # Use webcam
        use_webcam()
    
    else:
        print("Invalid choice")