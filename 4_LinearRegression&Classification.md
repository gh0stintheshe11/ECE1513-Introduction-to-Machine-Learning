# **Unit 4: Linear Regression and Classification**

---

## **1. General Idea**
- **Problem We Are Solving**: 
  - We need to **predict an output** based on given input data.
  - Two types of problems:
    - **Regression**: Predicting continuous values (e.g., predicting house prices).
    - **Classification**: Predicting discrete labels (e.g., spam vs. non-spam emails).

- **Models We Use**:
  - **Linear Regression**: Predicts **continuous values** using a **linear function** of input variables.
  - **Logistic Regression**: Predicts **binary classification** by applying a **sigmoid function** to linear regression.
  - **Gradient Descent**: Optimization algorithm used to find the best parameters for both regression and classification.

---

## **2. Important Definitions**
| **Term**                 | **Definition** |
|--------------------------|--------------|
| **Linear Regression** | A model that fits a straight line to data to predict continuous values. |
| **Hypothesis Function** | \( h(x) = w^T x + b \) (Linear function of input features). |
| **Loss Function (MSE)** | Measures the difference between predicted and actual values: \( J(w, b) = \frac{1}{N} \sum_{i=1}^{N} (h(x_i) - v_i)^2 \). |
| **Gradient Descent** | Optimization algorithm that updates parameters \( w, b \) iteratively to minimize the loss function. |
| **Learning Rate (\(\alpha\))** | Controls step size in gradient descent updates. |
| **Sigmoid Function** | Converts a linear regression output into a probability: \( \sigma(z) = \frac{1}{1 + e^{-z}} \). |
| **Cross-Entropy Loss** | Loss function for classification: \( J(w) = -\frac{1}{N} \sum_{i=1}^{N} \left[ v_i \log(\sigma(w^T x_i)) + (1 - v_i) \log(1 - \sigma(w^T x_i)) \right] \). |

---

## **3. Solution Process in Formulas**
### **1. Linear Regression**
#### **Step 1: Define Hypothesis**
\[
h(x) = w^T x + b
\]
where:
- \( x \) is the feature vector.
- \( w \) is the weight vector.
- \( b \) is the bias term.

#### **Step 2: Define Mean Squared Error (MSE) Loss Function**
\[
J(w, b) = \frac{1}{N} \sum_{i=1}^{N} (h(x_i) - v_i)^2
\]
where:
- \( N \) is the number of samples.
- \( v_i \) is the actual target value.

#### **Step 3: Compute Gradients**
\[
\frac{\partial J}{\partial w} = \frac{2}{N} \sum_{i=1}^{N} (h(x_i) - v_i) x_i
\]
\[
\frac{\partial J}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} (h(x_i) - v_i)
\]

#### **Step 4: Update Parameters Using Gradient Descent**
\[
w = w - \alpha \frac{\partial J}{\partial w}
\]
\[
b = b - \alpha \frac{\partial J}{\partial b}
\]
Repeat until convergence.

---

### **2. Logistic Regression (Binary Classification)**
#### **Step 1: Define Hypothesis (Apply Sigmoid)**
\[
P(v=1 | x; w) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}
\]

#### **Step 2: Define Cross-Entropy Loss Function**
\[
J(w) = -\frac{1}{N} \sum_{i=1}^{N} \left[ v_i \log(\sigma(w^T x_i)) + (1 - v_i) \log(1 - \sigma(w^T x_i)) \right]
\]

#### **Step 3: Compute Gradient**
\[
\frac{\partial J}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} (\sigma(w^T x_i) - v_i) x_i
\]

#### **Step 4: Update Parameters Using Gradient Descent**
\[
w = w - \alpha \frac{\partial J}{\partial w}
\]

Repeat until convergence.

---

## **4. Sample Numerical Example**
### **Example 1: Linear Regression**
#### **Given Data:**
| \( x \) | \( v \) (Actual) |
|--------|--------------|
| 1      | 2            |
| 2      | 2.5          |
| 3      | 3.5          |
| 4      | 4.5          |

#### **Step 1: Initialize Parameters**
Let \( w = 0 \), \( b = 0 \), \( \alpha = 0.1 \).

#### **Step 2: Compute Predictions**
\[
h(x) = w x + b
\]

#### **Step 3: Compute Loss (MSE)**
\[
J(w, b) = \frac{1}{4} [(w \cdot 1 + b - 2)^2 + (w \cdot 2 + b - 2.5)^2 + (w \cdot 3 + b - 3.5)^2 + (w \cdot 4 + b - 4.5)^2]
\]

#### **Step 4: Compute Gradients**
Using the gradient formulas, update \( w \) and \( b \) iteratively.

---

### **Example 2: Logistic Regression**
#### **Given Data:**
| \( x_1 \) | \( x_2 \) | \( v \) (Class) |
|--------|--------|--------------|
| 0.5    | 1.0    | 1            |
| 1.5    | 2.0    | 1            |
| 2.5    | 3.0    | 0            |
| 3.5    | 4.0    | 0            |

#### **Step 1: Initialize Parameters**
Let \( w = [0, 0] \), \( b = 0 \), \( \alpha = 0.1 \).

#### **Step 2: Compute Predictions Using Sigmoid**
\[
P(v=1 | x) = \frac{1}{1 + e^{-(w^T x + b)}}
\]

#### **Step 3: Compute Cross-Entropy Loss**
\[
J(w) = -\frac{1}{4} \sum_{i=1}^{4} \left[ v_i \log(\sigma(w^T x_i)) + (1 - v_i) \log(1 - \sigma(w^T x_i)) \right]
\]

#### **Step 4: Compute Gradients**
Using the gradient formulas, update \( w \) and \( b \) iteratively.

---

## **5. Others (Additional Important Details)**
- **Why Use MSE for Regression?**: Squared loss penalizes large errors more, making it a good choice for continuous predictions.
- **Why Use Cross-Entropy for Classification?**: It is better for probabilistic models like logistic regression because MSE leads to slow gradient updates.
- **Feature Scaling (Normalization)**: Before applying gradient descent, features should be normalized to improve convergence speed.