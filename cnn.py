# ============================================================
# Practical: Convolutional Neural Network (CNN)
# Problem  : Fashion Clothing Classification
# Dataset  : MNIST Fashion Dataset (built-in in Keras)
# ============================================================


# Cell 1 - Install libraries
# !pip install tensorflow matplotlib


# Cell 2 - Imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical


# Cell 3 - Load Dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',    'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',       'Ankle boot']

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)


# Cell 4 - Visualize Sample Images
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.suptitle("Sample Fashion Images")
plt.show()


# Cell 5 - Preprocess
# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# Reshape to add channel dimension (28, 28) -> (28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test,  10)

print("X_train shape after reshape:", X_train.shape)


# Cell 6 - Build & Compile CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10,  activation='softmax')   # 10 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Cell 7 - Train
history = model.fit(X_train, y_train, epochs=10, batch_size=64,
                    validation_split=0.1, verbose=1)


# Cell 8 - Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {acc * 100:.2f}%")


# Cell 9 - Accuracy & Loss Curves
plt.plot(history.history['accuracy'],     label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Cell 10 - Predict on Sample Test Images
y_pred = np.argmax(model.predict(X_test[:10]), axis=1)
y_true = np.argmax(y_test[:10], axis=1)

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    color = 'green' if y_pred[i] == y_true[i] else 'red'
    plt.title(class_names[y_pred[i]], color=color)
    plt.axis('off')
plt.suptitle("Predictions (Green=Correct, Red=Wrong)")
plt.show()

# Practical 4 — Convolutional Neural Network (CNN) using Fashion MNIST Classification

## Aim

To implement a Convolutional Neural Network (CNN) for image classification using the Fashion MNIST dataset in order to classify clothing items into different categories.

---

# Theory

# 1. Introduction to Image Classification

Image Classification is a computer vision task in which a machine learning model identifies and assigns a label to an image from predefined categories.

Examples:

* Cat vs Dog classification
* Face recognition
* Traffic sign detection
* Handwritten digit recognition
* Clothing classification

In this practical, the model classifies fashion product images into 10 clothing categories.

---

# 2. Convolutional Neural Network (CNN)

A Convolutional Neural Network (CNN) is a specialized type of Deep Neural Network designed specifically for processing image data.

Unlike traditional Dense Neural Networks:

* CNNs automatically learn spatial features
* Require fewer parameters
* Preserve image structure
* Detect patterns such as edges, textures, and shapes

CNNs are highly effective in:

* Computer Vision
* Image Recognition
* Object Detection
* Medical Imaging

---

# 3. Why CNN instead of Traditional Neural Networks?

In traditional fully connected networks:

* Every neuron connects to all pixels
* Number of parameters becomes very large
* Spatial relationships are lost

CNN overcomes these limitations by:

1. Local connectivity
2. Weight sharing
3. Feature extraction through convolution

This significantly reduces computational complexity.

---

# 4. Fashion MNIST Dataset

Fashion MNIST is a popular image classification dataset developed as a replacement for the original MNIST handwritten digit dataset.

It contains grayscale images of fashion products.

---

## Dataset Characteristics

| Property          | Value     |
| ----------------- | --------- |
| Total Images      | 70,000    |
| Training Images   | 60,000    |
| Testing Images    | 10,000    |
| Image Size        | 28 × 28   |
| Number of Classes | 10        |
| Image Type        | Grayscale |

---

## Fashion Categories

| Label | Category    |
| ----- | ----------- |
| 0     | T-shirt/Top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |

---

# 5. Image Representation

Images are represented as matrices of pixel values.

For grayscale images:

* Pixel range = 0 to 255
* 0 → Black
* 255 → White

Example:

A 28 × 28 image contains:

28 \times 28 = 784

pixel values.

---

# 6. Data Preprocessing

Preprocessing improves training efficiency and model performance.

---

## (a) Normalization

Pixel values are scaled between 0 and 1.

Formula:

x_{normalized} = \frac{x}{255}

Benefits:

* Faster convergence
* Stable gradient updates
* Improved numerical computation

---

## (b) Reshaping

CNN expects input images in tensor form.

For grayscale images:

(Height, Width, Channels) = (28,28,1)

Where:

* Height = 28
* Width = 28
* Channels = 1 (grayscale)

---

# 7. CNN Architecture

A typical CNN architecture contains:

1. Convolutional Layer
2. Activation Function
3. Pooling Layer
4. Flatten Layer
5. Dense Layers
6. Output Layer

---

# 8. Convolutional Layer

The Convolutional Layer is the core component of CNN.

It applies small learnable filters (kernels) over the image to detect features.

Examples of features detected:

* Edges
* Corners
* Textures
* Shapes

---

## Convolution Operation

Mathematically:

S(i,j)=(I*K)(i,j)=\sum_m\sum_n I(i-m,j-n)K(m,n)

Where:

* (I) = input image
* (K) = kernel/filter
* (S(i,j)) = output feature map

---

## Feature Maps

The output generated after convolution is called a Feature Map.

Feature maps highlight important image patterns.

---

## Advantages of Convolution

1. Automatic feature extraction
2. Parameter sharing
3. Translation invariance
4. Reduced computation

---

# 9. Filters / Kernels

Filters are small matrices that slide over the image.

Common filter sizes:

* 3 × 3
* 5 × 5

Each filter learns a different feature.

Example:

* Edge detector
* Vertical pattern detector
* Texture detector

---

# 10. Activation Function (ReLU)

After convolution, ReLU activation is applied.

Formula:

f(x)=\max(0,x)

Advantages:

* Faster training
* Avoids vanishing gradients
* Computationally efficient

---

# 11. Pooling Layer

Pooling reduces spatial dimensions of feature maps.

Most commonly used pooling:

* Max Pooling

---

## Max Pooling

Selects maximum value from a region.

Example:

2 × 2 pooling converts:

| Original | Pooled |
| -------- | ------ |
| 1 5      | 5      |
| 2 3      |        |

---

## Benefits of Pooling

1. Reduces computation
2. Prevents overfitting
3. Reduces feature dimensions
4. Retains important information

---

# 12. Flatten Layer

CNN layers produce multidimensional feature maps.

Flatten converts them into a one-dimensional vector.

Example:

(7,7,64) \rightarrow 3136

This vector is passed to Dense layers.

---

# 13. Dense (Fully Connected) Layer

Dense layers perform final classification.

Each neuron receives input from all previous neurons.

Operations:

1. Weighted sum
2. Bias addition
3. Activation function

---

## Weighted Sum

z = \sum_{i=1}^{n} w_i x_i + b

---

# 14. Output Layer

Fashion MNIST contains 10 classes.

Therefore:

* Output layer contains 10 neurons.

Softmax activation is used.

---

# 15. Softmax Activation Function

Softmax converts outputs into probability distribution.

Formula:

Softmax(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}

Where:

* (K) = number of classes

The class with highest probability is selected.

---

# 16. Forward Propagation in CNN

Steps:

1. Input image enters CNN
2. Convolution extracts features
3. ReLU introduces non-linearity
4. Pooling reduces dimensions
5. Flatten converts feature maps
6. Dense layers classify image
7. Softmax predicts probabilities

---

# 17. Loss Function

For multiclass image classification:

Categorical Cross-Entropy Loss is used.

Formula:

L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)

Where:

* (y_i) = actual label
* (\hat{y}_i) = predicted probability

Lower loss indicates better performance.

---

# 18. Backpropagation

Backpropagation computes gradients and updates CNN weights.

---

## Weight Update Formula

w_{new}=w_{old}-\eta\frac{\partial L}{\partial w}

Where:

* (\eta) = learning rate

---

# 19. Optimizer

Common optimizers:

1. SGD
2. RMSProp
3. Adam

Adam optimizer is widely preferred because:

* Faster convergence
* Adaptive learning rates
* Stable training

---

# 20. Evaluation Metrics

---

## (a) Accuracy

Measures correctly classified images.

Formula:

Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}

---

## (b) Confusion Matrix

Shows actual vs predicted classes.

Useful for identifying:

* Misclassified clothing items
* Similar categories causing confusion

---

## (c) Precision

Precision = \frac{TP}{TP+FP}

---

## (d) Recall

Recall = \frac{TP}{TP+FN}

---

## (e) F1-Score

F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}

---

# 21. Overfitting in CNN

Overfitting occurs when the model performs well on training data but poorly on unseen data.

---

## Prevention Techniques

1. Dropout Layer
2. Data Augmentation
3. Regularization
4. Early Stopping

---

# 22. Dropout Layer

Dropout randomly disables neurons during training.

Advantages:

* Reduces overfitting
* Improves generalization
* Prevents neuron dependency

---

# 23. Data Augmentation

Artificially increases dataset size by modifying images.

Techniques:

1. Rotation
2. Flipping
3. Zooming
4. Shifting

Benefits:

* Improves robustness
* Reduces overfitting
* Enhances generalization

---

# 24. Applications of CNN

1. Face Recognition
2. Medical Image Analysis
3. Self-driving Cars
4. Object Detection
5. Surveillance Systems
6. Fingerprint Recognition
7. Fashion Recommendation Systems

---

# 25. Advantages of CNN

1. Automatic feature extraction
2. High image classification accuracy
3. Efficient parameter sharing
4. Handles large image datasets
5. Learns spatial hierarchies effectively

---

# 26. Conclusion

Convolutional Neural Networks are highly effective for image classification tasks. In this practical, the Fashion MNIST dataset is used to classify grayscale images of clothing items into 10 categories. Convolutional layers automatically extract image features, while pooling layers reduce dimensions and improve efficiency. Dense layers perform final classification using Softmax activation. The model is trained using categorical cross-entropy loss and evaluated using metrics such as accuracy, precision, recall, and F1-score. This practical demonstrates the power of CNNs in computer vision and image recognition applications.
