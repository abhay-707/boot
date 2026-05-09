# ============================================================
# Practical: Classification using Deep Neural Network
# Problem  : Multiclass Classification - OCR Letter Recognition
# Dataset  : UCI Letter Recognition Dataset
# ============================================================


# Cell 1 - Install libraries
# !pip install numpy pandas matplotlib scikit-learn tensorflow


# Cell 2 - Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical


# Cell 3 - Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
df = pd.read_csv(url, header=None)

X = df.iloc[:, 1:].values   # 16 features
y = df.iloc[:, 0].values    # letters A-Z

print("Shape:", X.shape)
print("Classes:", np.unique(y))


# Cell 4 - Preprocess
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot  = to_categorical(y_encoded, num_classes=26)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2,
                                                     random_state=42)
_, _, y_train_int, y_test_int = train_test_split(X, y_encoded, test_size=0.2,
                                                  random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# Cell 5 - Build & Compile Model
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64,  activation='relu'),
    layers.Dense(26,  activation='softmax')   # 26 classes A-Z
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Cell 6 - Train
history = model.fit(X_train, y_train, epochs=50, batch_size=64,
                    validation_split=0.1, verbose=1)


# Cell 7 - Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test_int, y_pred, target_names=le.classes_))


# Cell 8 - Loss & Accuracy Curves
plt.plot(history.history['accuracy'],     label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Cell 9 - Confusion Matrix
cm = confusion_matrix(y_test_int, y_pred)
plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(26), le.classes_)
plt.yticks(range(26), le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

'''
# Practical 2 — Multiclass Classification using Deep Neural Network (OCR Letter Recognition)

## Aim

To implement a Deep Neural Network (DNN) for multiclass classification using the OCR Letter Recognition dataset in order to identify capital English alphabets (A–Z) based on extracted image features.

---

# Theory

# 1. Introduction to Classification

Classification is a supervised machine learning technique used to assign input data into predefined categories or classes.

Unlike regression, where the output is continuous, classification predicts discrete labels.

Examples:

* Email Spam Detection → Spam / Not Spam
* Disease Prediction → Positive / Negative
* Digit Recognition → 0–9 classes
* Letter Recognition → A–Z classes

---

# 2. Multiclass Classification

Multiclass classification refers to classification problems where the output can belong to more than two classes.

In this practical:

* Total Classes = 26
* Classes represent capital letters:
  A, B, C, ..., Z

The model predicts the probability of the input belonging to each class and selects the class with the highest probability.

---

# 3. Deep Neural Network (DNN)

A Deep Neural Network is an Artificial Neural Network containing multiple hidden layers between input and output layers.

The network learns patterns through:

1. Forward Propagation
2. Loss Computation
3. Backpropagation
4. Weight Optimization

A DNN can learn highly complex decision boundaries for classification problems.

---

# 4. Architecture of Neural Network

The DNN architecture for multiclass classification consists of:

1. Input Layer
2. Hidden Layers
3. Output Layer

---

## (a) Input Layer

The OCR Letter Recognition dataset contains 16 numerical features extracted from character images.

Therefore:

* Number of input neurons = 16

These features represent:

* Pixel statistics
* Edge counts
* Geometric characteristics
* Image shape information

---

## (b) Hidden Layers

Hidden layers extract meaningful representations from input data.

Each neuron performs:

### Weighted Sum

z = \sum_{i=1}^{n} w_i x_i + b

Where:

* (x_i) = input feature
* (w_i) = weight
* (b) = bias

---

### Activation Function

ReLU activation is commonly used.

f(x)=\max(0,x)

### Advantages of ReLU

* Faster computation
* Avoids vanishing gradient problem
* Efficient learning in deep networks

---

# 5. Output Layer

For multiclass classification:

* One neuron is used for each class.

Since there are 26 letters:

* Output layer contains 26 neurons.

The output layer uses the Softmax activation function.

---

# 6. Softmax Activation Function

Softmax converts raw outputs into probability values between 0 and 1.

The probabilities of all classes sum to 1.

Formula:

Softmax(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}

Where:

* (z_i) = output score of class (i)
* (K) = total number of classes

---

## Example

Suppose output probabilities are:

| Letter | Probability |
| ------ | ----------- |
| A      | 0.05        |
| B      | 0.02        |
| C      | 0.90        |
| D      | 0.03        |

The model predicts class **C** because it has the highest probability.

---

# 7. OCR Letter Recognition Dataset

OCR (Optical Character Recognition) is a technology used to identify text characters from images.

The UCI OCR Letter Recognition dataset contains extracted numerical features of English capital letters.

---

## Dataset Characteristics

| Property           | Value  |
| ------------------ | ------ |
| Total Samples      | 20,000 |
| Number of Features | 16     |
| Number of Classes  | 26     |
| Output Labels      | A–Z    |

---

## Applications of OCR

1. Document digitization
2. Handwritten text recognition
3. Bank cheque processing
4. Postal code recognition
5. Automatic number plate recognition

---

# 8. Data Preprocessing

Data preprocessing is essential for improving model performance.

---

## (a) Feature Scaling

Input features may have different ranges.

Standardization formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"z = \frac{x-\mu}{\sigma}"}}

Where:

* (x) = original value
* (\mu) = mean
* (\sigma) = standard deviation

Benefits:

* Faster convergence
* Stable learning
* Prevents feature dominance

---

## (b) Label Encoding

Machine learning models cannot directly process alphabet labels.

Therefore:

| Letter | Encoded Value |
| ------ | ------------- |
| A      | 0             |
| B      | 1             |
| C      | 2             |

---

## (c) One-Hot Encoding

For multiclass classification, labels are converted into binary vectors.

Example:

| Letter | One-Hot Vector |
| ------ | -------------- |
| A      | [1,0,0,0,...]  |
| B      | [0,1,0,0,...]  |
| C      | [0,0,1,0,...]  |

This representation is suitable for Softmax classification.

---

# 9. Forward Propagation

Forward propagation passes inputs through the network layer by layer.

Steps:

1. Input features enter network
2. Weighted sums are calculated
3. Activation functions applied
4. Softmax produces class probabilities
5. Predicted class selected

---

# 10. Loss Function

The model learns by minimizing prediction error.

For multiclass classification, Categorical Cross-Entropy Loss is used.

Formula:

L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)

Where:

* (y_i) = actual class value
* (\hat{y}_i) = predicted probability
* (C) = number of classes

---

## Purpose of Cross-Entropy

* Penalizes incorrect predictions
* Encourages higher probability for correct class
* Suitable for probability outputs

Lower loss indicates better predictions.

---

# 11. Backpropagation

Backpropagation computes gradients of loss with respect to network weights.

Weights are updated using Gradient Descent.

---

## Weight Update Formula

w_{new}=w_{old}-\eta\frac{\partial L}{\partial w}

Where:

* (\eta) = learning rate
* (L) = loss function

---

# 12. Optimizers

Optimizers improve training speed and convergence.

Common optimizers:

1. SGD
2. RMSProp
3. Adam

Adam optimizer is most commonly used because it:

* Adapts learning rate automatically
* Converges faster
* Provides stable training

---

# 13. Evaluation Metrics

---

## (a) Accuracy

Accuracy measures the percentage of correctly classified samples.

Formula:

Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}

Higher accuracy indicates better model performance.

---

## (b) Confusion Matrix

A confusion matrix shows actual vs predicted classes.

It helps identify:

* Correct classifications
* Misclassifications
* Frequently confused letters

---

## (c) Precision

Measures correctness of positive predictions.

Precision = \frac{TP}{TP+FP}

---

## (d) Recall

Measures ability to identify actual positives.

Recall = \frac{TP}{TP+FN}

---

## (e) F1-Score

Harmonic mean of precision and recall.

F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}

---

# 14. Epochs and Batch Size

## Epoch

One complete iteration over the entire dataset.

## Batch Size

Number of samples processed before weight updates.

Common types:

* Batch Gradient Descent
* Stochastic Gradient Descent
* Mini-Batch Gradient Descent

Mini-batch is widely preferred.

---

# 15. Overfitting and Regularization

---

## Overfitting

Occurs when the model memorizes training data but fails on unseen data.

### Prevention Techniques

1. Dropout
2. Early Stopping
3. L2 Regularization
4. More training data

---

## Dropout

Randomly disables neurons during training.

Advantages:

* Prevents co-adaptation
* Improves generalization
* Reduces overfitting

---

# 16. Advantages of DNN for Classification

1. Learns complex feature relationships
2. High classification accuracy
3. Automatic feature extraction
4. Handles large datasets
5. Suitable for image and pattern recognition

---

# 17. Applications of OCR Classification

1. Handwriting recognition
2. Automatic form reading
3. CAPTCHA solving
4. Passport scanning
5. Digitization of books and documents
6. Postal sorting systems

---

# 18. Conclusion

Deep Neural Networks are highly effective for multiclass classification problems. In this practical, the OCR Letter Recognition dataset is used to classify 26 English capital letters using a DNN model. Hidden layers with ReLU activation learn important feature patterns, while the Softmax output layer predicts class probabilities. The model is trained using categorical cross-entropy loss and evaluated using accuracy, precision, recall, and F1-score. This practical demonstrates how neural networks can successfully perform large-scale pattern recognition tasks.

'''