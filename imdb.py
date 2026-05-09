# ============================================================
# Practical: Binary Classification using Deep Neural Network
# Problem  : IMDB Movie Review Sentiment (Positive / Negative)
# Dataset  : IMDB dataset (built-in in Keras)
# ============================================================


# Cell 1 - Install libraries
# !pip install tensorflow


# Cell 2 - Imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Cell 3 - Load Dataset
# Load top 10000 most frequent words only
NUM_WORDS = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

print("Training samples :", len(X_train))
print("Test samples     :", len(X_test))
print("Sample review (encoded):", X_train[0][:10])
print("Label (0=neg, 1=pos)   :", y_train[0])


# Cell 4 - Preprocess (Pad Sequences to same length)
MAX_LEN = 200

X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_test  = pad_sequences(X_test,  maxlen=MAX_LEN)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)


# Cell 5 - Build & Compile Model
model = models.Sequential([
    layers.Embedding(NUM_WORDS, 32, input_length=MAX_LEN),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1,  activation='sigmoid')   # binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# Cell 6 - Train
history = model.fit(X_train, y_train, epochs=10, batch_size=128,
                    validation_split=0.2, verbose=1)


# Cell 7 - Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {acc * 100:.2f}%")


# Cell 8 - Accuracy & Loss Curves
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


# Cell 9 - Predict on Sample Reviews
y_pred = (model.predict(X_test[:10]) > 0.5).astype(int).flatten()

for i in range(10):
    actual    = "Positive" if y_test[i]  == 1 else "Negative"
    predicted = "Positive" if y_pred[i]  == 1 else "Negative"
    print(f"Review {i+1} | Actual: {actual:8} | Predicted: {predicted}")

    '''
    # Practical 3 — Binary Classification using Deep Neural Network (IMDB Sentiment Analysis)

## Aim

To implement a Deep Neural Network (DNN) for binary classification using the IMDB movie review dataset in order to classify reviews as positive or negative sentiments.

---

# Theory

# 1. Introduction to Binary Classification

Binary Classification is a supervised machine learning task in which the output belongs to one of two possible classes.

Examples:

* Spam / Not Spam
* Fraud / Genuine
* Positive / Negative
* Disease / No Disease

In this practical, the task is Sentiment Analysis, where movie reviews are classified as:

* Positive Sentiment
* Negative Sentiment

---

# 2. Sentiment Analysis

Sentiment Analysis is a Natural Language Processing (NLP) technique used to determine emotional tone or opinion from text data.

The goal is to identify whether a text expresses:

* Positive emotion
* Negative emotion
* Neutral emotion

Examples:

| Review                                | Sentiment |
| ------------------------------------- | --------- |
| “Amazing movie with excellent acting” | Positive  |
| “Worst movie I have ever watched”     | Negative  |

---

# 3. Natural Language Processing (NLP)

Natural Language Processing is a branch of Artificial Intelligence that enables computers to understand, interpret, and process human language.

NLP tasks include:

1. Text Classification
2. Machine Translation
3. Chatbots
4. Speech Recognition
5. Sentiment Analysis
6. Text Summarization

In this practical, NLP techniques are used to process movie review text data.

---

# 4. IMDB Dataset

The IMDB Movie Review dataset is a popular benchmark dataset for sentiment analysis.

---

## Dataset Characteristics

| Property            | Value  |
| ------------------- | ------ |
| Total Reviews       | 50,000 |
| Positive Reviews    | 25,000 |
| Negative Reviews    | 25,000 |
| Data Type           | Text   |
| Classification Type | Binary |

---

## Dataset Split

| Dataset       | Samples |
| ------------- | ------- |
| Training Data | 25,000  |
| Testing Data  | 25,000  |

---

# 5. Text Representation

Computers cannot directly understand raw text.

Therefore, text must be converted into numerical format before training.

---

# 6. Tokenization

Tokenization is the process of splitting text into smaller units called tokens.

Example:

Sentence:

> “This movie is excellent”

Tokens:

> [“This”, “movie”, “is”, “excellent”]

In the IMDB dataset:

* Words are converted into integer indices.

Example:

| Word      | Index |
| --------- | ----- |
| movie     | 15    |
| excellent | 83    |
| bad       | 45    |

Thus, a review becomes a sequence of integers.

---

# 7. Vocabulary Size

Vocabulary refers to the total number of unique words considered by the model.

Example:

* Top 10,000 most frequent words may be used.

Rare words are usually ignored to:

* Reduce complexity
* Improve efficiency
* Prevent overfitting

---

# 8. Sequence Padding

Movie reviews have different lengths.

Neural networks require fixed-size input sequences.

Padding adds zeros to shorter sequences.

Example:

Original:

> [12, 45, 83]

Padded:

> [12, 45, 83, 0, 0, 0]

Benefits:

* Uniform input size
* Efficient batch processing

---

# 9. Embedding Layer

The Embedding Layer converts integer word indices into dense numerical vectors.

Instead of sparse representations, embeddings capture semantic meaning of words.

Example:

| Word      | Embedding Vector   |
| --------- | ------------------ |
| good      | [0.24, 0.88, 0.12] |
| excellent | [0.26, 0.90, 0.15] |

Words with similar meanings have similar vector representations.

---

## Embedding Representation

Embedding: \mathbb{R}^{V} \rightarrow \mathbb{R}^{d}

Where:

* (V) = vocabulary size
* (d) = embedding dimension

---

## Advantages of Embeddings

1. Captures semantic relationships
2. Reduces dimensionality
3. Improves learning efficiency
4. Better representation of language

---

# 10. Deep Neural Network Architecture

The DNN architecture typically contains:

1. Embedding Layer
2. Dense Hidden Layers
3. Output Layer

---

## (a) Hidden Layers

Hidden layers learn sentiment-related patterns.

Each neuron performs:

### Weighted Sum

z = \sum_{i=1}^{n} w_i x_i + b

---

### Activation Function (ReLU)

f(x)=\max(0,x)

Advantages:

* Faster computation
* Prevents vanishing gradients
* Efficient deep learning

---

# 11. Output Layer

Binary classification requires only one output neuron.

The output layer uses the Sigmoid activation function.

---

# 12. Sigmoid Activation Function

Sigmoid converts output into a probability value between 0 and 1.

Formula:

\sigma(x)=\frac{1}{1+e^{-x}}

---

## Interpretation

| Output Probability | Predicted Sentiment |
| ------------------ | ------------------- |
| > 0.5              | Positive            |
| < 0.5              | Negative            |

Example:

* Output = 0.92 → Positive Review
* Output = 0.12 → Negative Review

---

# 13. Forward Propagation

Forward propagation involves passing input text through the network.

Steps:

1. Convert words into embeddings
2. Process embeddings through hidden layers
3. Generate output probability using Sigmoid
4. Predict sentiment class

---

# 14. Loss Function

For binary classification, Binary Cross-Entropy Loss is used.

Formula:

L = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(\hat{y}_i)+(1-y_i)\log(1-\hat{y}_i)\right]

Where:

* (y_i) = actual label
* (\hat{y}_i) = predicted probability
* (N) = total samples

---

## Purpose of Binary Cross-Entropy

* Penalizes incorrect predictions
* Encourages accurate probabilities
* Suitable for binary outputs

Lower loss indicates better performance.

---

# 15. Backpropagation

Backpropagation calculates gradients of loss and updates weights.

---

## Weight Update Formula

w_{new}=w_{old}-\eta\frac{\partial L}{\partial w}

Where:

* (\eta) = learning rate
* (L) = loss function

---

# 16. Optimizer

Optimizers improve learning efficiency.

Common optimizers:

1. SGD
2. RMSProp
3. Adam

Adam optimizer is most commonly used because it:

* Adapts learning rates automatically
* Converges quickly
* Provides stable training

---

# 17. Evaluation Metrics

---

## (a) Accuracy

Measures percentage of correct predictions.

Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}

---

## (b) Precision

Measures correctness of positive predictions.

Precision = \frac{TP}{TP+FP}

---

## (c) Recall

Measures ability to identify actual positives.

Recall = \frac{TP}{TP+FN}

---

## (d) F1-Score

Harmonic mean of precision and recall.

F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}

---

## (e) Confusion Matrix

A confusion matrix displays:

* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

It helps analyze classification errors.

---

# 18. Overfitting in NLP Models

Overfitting occurs when the model memorizes training data.

---

## Prevention Techniques

1. Dropout Layer
2. Early Stopping
3. Regularization
4. More Training Data

---

# 19. Dropout Layer

Dropout randomly disables neurons during training.

Advantages:

* Reduces overfitting
* Improves generalization
* Prevents dependency between neurons

---

# 20. Applications of Sentiment Analysis

1. Social media monitoring
2. Product review analysis
3. Customer feedback analysis
4. Political opinion mining
5. Brand reputation management
6. Chatbots and recommendation systems

---

# 21. Advantages of Deep Learning in NLP

1. Learns semantic relationships automatically
2. Handles large text datasets
3. Improves prediction accuracy
4. Captures contextual meaning
5. Reduces manual feature engineering

---

# 22. Conclusion

Deep Neural Networks are highly effective for binary classification tasks such as sentiment analysis. In this practical, the IMDB movie review dataset is used to classify reviews as positive or negative sentiments. Text data is converted into numerical form using tokenization and embedding layers. Hidden layers learn sentiment-related patterns, while the Sigmoid output layer predicts probabilities for binary classification. The model is trained using binary cross-entropy loss and evaluated using metrics such as accuracy, precision, recall, and F1-score. This practical demonstrates the effectiveness of deep learning techniques in Natural Language Processing applications.
'''