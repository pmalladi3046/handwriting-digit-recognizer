# handwriting-digit-recognizer
This is a handwriting digit recognition program that uses Deep Learning (Convolutional Neural Networks) with PyTorch to classify images
of handwritten digits (0–9). The model is trained on the MNIST dataset, achieving ~98–99% accuracy, and includes an interactive mode that
allows users to upload their own handwritten digit images for real-time identification.

Included Files:

    - scanner.py: The main Python script containing:
        - CNN model definition
        - Training, evaluation, and visualization
        - Interactive user input loop for custom digit images
    - data: Automatically created by PyTorch when downloading the MNIST dataset

User Requirements:
    
    - Python 3.6 or newer
    - PyTorch (torch, torchvision)
    - Matplotlib
    - Pillow (PIL)

Features:

    - Trains on the MNIST dataset consisting of 60,000 training images and 10,000 test images of handwritten digits (28×28 pixels, grayscale).
    - Convolutional Neural Network (CNN) with two convolutional layers, pooling, and fully connected layers for high-accuracy digit recognition.
    - Evaluates model performance on unseen test data and prints accuracy.
    - Visualizes predictions for sample test images using Matplotlib.
    - Interactive user input loop:
        - Users can upload their own handwritten digit images (.png, .jpg, etc.)
        - The model predicts the digit and displays the image with the predicted label
        - Infinite loop until the user types "quit"

Structure of the Program:

    - Section 0: Import all required libraries
        - PyTorch (torch, torch.nn, torch.optim)
        - torchvision (datasets, transforms)
        - Matplotlib and Pillow

    - Section 1: Load and preprocess the MNIST dataset
        - Normalize pixel values between -1 and 1
        -Convert images to PyTorch tensors

    - Section 2: Define CNN architecture
        - Two convolutional layers
        - Max pooling
        - Two fully connected layers
        - ReLU activation functions

    - Section 3: Train the model
        - Use Cross-Entropy Loss
        - Adam optimizer
        - Training loop over multiple epochs

    - Section 4: Evaluate model
        - Compute test accuracy
        - Print evaluation metrics

    - Section 5: Visualize predictions
        - Display 5 test images with their true labels and predicted digits

    - Section 6: Interactive user input
        - Users upload custom digit images
        - Images are resized to 28×28, converted to grayscale, normalized, and fed to the model
        - Prediction displayed with Matplotlib

Efficiency:
    * m = number of images, n = number of pixels per image, f = number of filters/neurons
    - Data Preprocessing:
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)

     - CNN Forward Pass:
        Time Complexity: O(m * n * f)
        Space Complexity: O(m * n * f)
    
     - Model Training:
        Time Complexity: O(epochs * m * n * f)
        Space Complexity: O(weights + activations)
    
     - Prediction:
        Time Complexity: O(n * f)
        Space Complexity: O(weights)

How to Run:

    - run the program with the command: python scanner.py
    - after model training is finished, either:
        - type the file path to a handwritten digit image to get a prediction
        or
        - type "quit" to exit the program 