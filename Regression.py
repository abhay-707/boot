# ============================================================
# Practical: Linear Regression using Deep Neural Network
# Problem  : Boston Housing Price Prediction
# ============================================================


# Cell 1 - Install libraries
# !pip install numpy pandas matplotlib scikit-learn tensorflow


# Cell 2 - Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from tensorflow.keras import layers, models


# Cell 3 - Load Dataset
# Note: load_boston() removed in sklearn v1.2+, using URL workaround
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

print("Shape:", X.shape)


# Cell 4 - Preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# Cell 5 - Build & Compile Model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)   # linear output for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()


# Cell 6 - Train
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.1, verbose=1)


# Cell 7 - Evaluate
y_pred = model.predict(X_test).flatten()

print("MSE  :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2   :", r2_score(y_test, y_pred))


# Cell 8 - Loss Curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.show()


# Cell 9 - Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


'''
# Practical 1 — Linear Regression using Deep Neural Network (Boston Housing Dataset)

## Aim

To implement a Deep Neural Network (DNN) model for performing linear regression on the Boston Housing dataset in order to predict median house prices based on multiple input features.

---

# Theory

## 1. Introduction to Linear Regression

Linear Regression is a supervised machine learning algorithm used for predicting continuous numerical values. The objective of regression is to establish a relationship between independent variables (features) and a dependent variable (target).

In this practical, the target variable is the median price of houses, while the input variables include various housing-related features such as:

* Crime rate
* Number of rooms
* Nitric oxide concentration
* Property tax rate
* Accessibility to highways
* Student-teacher ratio

Traditional linear regression attempts to fit a straight-line equation to the data.

The mathematical representation of linear regression is:

y = w_1x_1 + w_2x_2 + w_3x_3 + \cdots + w_nx_n + b

Where:

* (y) = predicted output
* (x_1, x_2, ..., x_n) = input features
* (w_1, w_2, ..., w_n) = weights
* (b) = bias term

The model learns the optimal weights and bias values during training.

---

# 2. Deep Neural Network for Regression

A Deep Neural Network (DNN) is an advanced machine learning model inspired by the structure of the human brain. It consists of interconnected layers of artificial neurons.

A DNN generally contains:

1. Input Layer
2. Hidden Layers
3. Output Layer

Unlike classical linear regression, a DNN can learn both linear and non-linear relationships in the dataset.

For regression problems:

* Hidden layers generally use the ReLU activation function.
* Output layer uses a Linear activation function because the output is a continuous value.

---

# 3. Structure of Neural Network

## (a) Input Layer

The input layer receives feature values from the dataset.

In the Boston Housing dataset:

* Total features = 13
* Therefore, the input layer contains 13 neurons.

---

## (b) Hidden Layers

Hidden layers perform feature extraction and pattern learning.

Each neuron performs two operations:

### Step 1: Weighted Sum

z = \sum_{i=1}^{n} w_i x_i + b

### Step 2: Activation Function

ReLU (Rectified Linear Unit) activation is commonly used.

f(x) = \max(0,x)

### Advantages of ReLU

* Faster training
* Reduces vanishing gradient problem
* Computationally efficient

---

## (c) Output Layer

The output layer contains a single neuron because only one numerical value (house price) is predicted.

The activation function is linear:

genui{"math_block_widget_always_prefetch_v2":{"content":"f(x)=x"}}

This allows the network to predict any real-valued output.

---

# 4. Boston Housing Dataset

The Boston Housing dataset is a famous regression dataset used for machine learning experiments.

## Dataset Characteristics

| Property      | Value              |
| ------------- | ------------------ |
| Total Samples | 506                |
| Features      | 13                 |
| Output        | Median House Price |
| Data Type     | Numerical          |

---

## Important Features

| Feature | Description                           |
| ------- | ------------------------------------- |
| CRIM    | Crime rate                            |
| RM      | Average number of rooms               |
| NOX     | Nitric oxide concentration            |
| TAX     | Property tax rate                     |
| PTRATIO | Student-teacher ratio                 |
| LSTAT   | Percentage of lower status population |

---

# 5. Data Preprocessing

Data preprocessing improves model performance and stability.

## Steps involved:

### (a) Data Normalization / Standardization

Features may have different scales. Standardization transforms features into a common range.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"z = \frac{x-\mu}{\sigma}"}}

Where:

* (x) = original value
* (\mu) = mean
* (\sigma) = standard deviation

### Benefits

* Faster convergence
* Better gradient descent performance
* Prevents domination by large-valued features

---

## (b) Train-Test Split

The dataset is divided into:

* Training Data → Used for learning
* Testing Data → Used for evaluation

Typical split:

* 80% Training
* 20% Testing

---

# 6. Forward Propagation

Forward propagation is the process of passing inputs through the network to generate predictions.

Steps:

1. Input features are multiplied by weights.
2. Bias is added.
3. Activation function is applied.
4. Output is passed to next layer.
5. Final prediction is produced.

---

# 7. Loss Function

The network learns by minimizing prediction error.

For regression problems, Mean Squared Error (MSE) is commonly used.

## Mean Squared Error (MSE)

MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2

Where:

* (y_i) = actual value
* (\hat{y}_i) = predicted value
* (n) = number of samples

### Characteristics of MSE

* Penalizes large errors heavily
* Differentiable and easy to optimize
* Lower MSE indicates better performance

---

# 8. Backpropagation

Backpropagation is the learning mechanism of neural networks.

It calculates gradients of the loss function with respect to weights and updates them using Gradient Descent.

## Weight Update Formula

w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}

Where:

* (\eta) = learning rate
* (L) = loss function
* (\frac{\partial L}{\partial w}) = gradient

---

# 9. Optimizer

Optimizers improve training efficiency.

Common optimizers:

* SGD (Stochastic Gradient Descent)
* RMSProp
* Adam

Adam optimizer is widely used because it combines:

* Momentum
* Adaptive learning rates

Advantages:

* Faster convergence
* Stable training
* Efficient for deep networks

---

# 10. Evaluation Metrics

The performance of regression models is measured using several metrics.

---

## (a) Mean Squared Error (MSE)

Measures average squared prediction error.

Lower value indicates better performance.

---

## (b) Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

RMSE = \sqrt{MSE}

Advantages:

* Same unit as target variable
* Easier interpretation

---

## (c) R² Score (Coefficient of Determination)

Measures how well the model explains variance in data.

R^2 = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}

Where:

* (\bar{y}) = mean of actual values

### Interpretation

| R² Value | Meaning             |
| -------- | ------------------- |
| 1        | Perfect prediction  |
| 0        | No predictive power |
| Negative | Poor model          |

---

# 11. Epochs and Batch Size

## Epoch

One complete pass through the entire training dataset.

## Batch Size

Number of samples processed before updating weights.

### Types

* Batch Gradient Descent
* Mini-Batch Gradient Descent
* Stochastic Gradient Descent

Mini-batch is commonly preferred.

---

# 12. Overfitting and Underfitting

## Overfitting

Model memorizes training data but performs poorly on new data.

### Prevention Methods

* Dropout
* Regularization
* Early stopping

---

## Underfitting

Model fails to learn underlying patterns.

Causes:

* Insufficient layers
* Low training time
* Simple model

---

# 13. Advantages of DNN-based Regression

1. Learns complex non-linear relationships
2. Handles large datasets efficiently
3. Automatic feature learning
4. Better prediction accuracy
5. Scalable to real-world applications

---

# 14. Applications of Regression using DNN

1. House price prediction
2. Stock market forecasting
3. Weather forecasting
4. Sales prediction
5. Medical analysis
6. Energy consumption prediction

---

# 15. Conclusion

Deep Neural Networks can effectively perform regression tasks by learning relationships between input features and continuous outputs. In this practical, the Boston Housing dataset is used to predict house prices using multiple numerical features. Hidden layers with ReLU activation help the model learn complex patterns, while the output layer predicts continuous values. Performance is evaluated using MSE, RMSE, and R² score, demonstrating the effectiveness of neural networks for regression problems.
'''