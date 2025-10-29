# Task3-Number-Recognition-Bharat-intern-Data-Science

https://github.com/Premkumar9799817360/Task3-Number-Recognition-Bharat-intern-Data-Science/assets/83695512/943ed54d-25ff-41ef-be59-529abe1c5291



## 🧠 Project Overview
The **Handwritten Digit Recognition System** is a deep learning-based project developed using the **MNIST dataset**.  
This model identifies digits (0–9) from scanned images of handwritten numbers using an **Artificial Neural Network (ANN)** built with **Keras**.

This task demonstrates a strong understanding of **neural networks, image preprocessing, model training, and evaluation** — a core skill in computer vision and AI.

---

## 🎯 Objective
To build a **Neural Network model** capable of recognizing handwritten digits from the MNIST dataset with high accuracy.

---

## 📊 Dataset Description – MNIST
The **MNIST dataset** consists of:
- **60,000 training images** and **10,000 testing images**
- Each image is **28x28 pixels**, grayscale (0–255 intensity)
- Labels correspond to digits **0–9**

It serves as a standard benchmark for handwritten digit recognition tasks in machine learning.

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading
The MNIST dataset is loaded directly from the Keras library.

```python
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train → Training data and labels

X_test, y_test → Testing data and labels

2️⃣ Importing Libraries
python
Copy code
# Importing essential libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import mnist
import keras
These libraries are used for:

Data handling: NumPy, Pandas

Visualization: Matplotlib

Deep learning: Keras

3️⃣ Model Compilation
python
Copy code
# Model compilation
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=["accuracy"]
)
Loss Function: categorical_crossentropy — used for multi-class classification

Optimizer: RMSprop() — optimizes gradient descent

Metric: Accuracy

4️⃣ Model Training
python
Copy code
# Training input/output variables, batch size, epochs
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2
)
The model is trained over defined epochs using the given batch size.

Training continues until convergence with minimal loss and high accuracy.

5️⃣ Model Evaluation
python
Copy code
print("[INFO] Calculating model accuracy")
scores = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")
Output Example:

css
Copy code
[INFO] Calculating model accuracy
313/313 [==============================] - 2s 4ms/step - loss: 0.1343 - accuracy: 0.9645
Test Accuracy: 96.45%
The model achieves an impressive 96.45% accuracy on unseen test data.

6️⃣ Model Performance Summary
python
Copy code
# Evaluate test loss and accuracy
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
Final Output:

yaml
Copy code
Test loss: 0.1343
Test accuracy: 0.9645
This demonstrates the model’s excellent generalization ability on new handwritten digit images.

7️⃣ Saving the Model
python
Copy code
# Save model and architecture to single file
model.save('MNIST-ANN.h5')
model.summary()
print("Saved model")
The trained model and its architecture are saved as a single .h5 file for future reuse or deployment.

📈 Model Summary
The Sequential ANN model used for handwritten digit classification includes:

Input Layer: Takes 28×28 pixel grayscale image data

Hidden Layers: Fully connected Dense layers with activation functions

Output Layer: 10 neurons (for digits 0–9) using Softmax activation

🧩 Tools & Technologies Used
Category	Tools / Libraries
Programming	Python
Data Handling	NumPy, Pandas
Visualization	Matplotlib
Deep Learning Framework	Keras
Optimizer	RMSprop
Model Type	Artificial Neural Network (ANN)
Dataset	MNIST (Handwritten Digits)

🎯 Results
Metric	Value
Test Loss	0.1343
Test Accuracy	96.45% ✅

The model accurately recognizes handwritten digits, demonstrating strong feature learning and generalization capabilities.

🧾 Conclusion
The Number Recognition System successfully detects handwritten digits using a deep learning model trained on the MNIST dataset.
Through this task, I gained practical experience in:

Neural network design

Model compilation and optimization

Evaluating model performance

Saving and deploying trained models

📁 Project Name: Task3-Number-Recognition-Bharat-intern-Data-Science
💼 Internship: Bharat Intern – Data Science
🧠 Task: Handwritten Digit Recognition using ANN
🎯 Accuracy Achieved: 96.45%

Copy code








