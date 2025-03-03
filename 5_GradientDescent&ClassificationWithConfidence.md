# **Unit 5: Gradient Descent and Classification with Confidence**

---

## **1. General Idea**
- **Problem We Are Solving**:  
  - We aim to **optimize models** using **gradient descent**.
  - We also want to **classify data with confidence scores** (i.e., measure how sure the model is about its predictions).

- **Models We Use**:  
  - **Gradient Descent**: Optimizes functions by iteratively adjusting parameters in the direction of the steepest descent.
  - **Logistic Regression**: Used for binary classification, where confidence is derived from the **sigmoid function** output.
  - **Softmax Regression**: Used for multi-class classification, converting raw scores into probabilities.

---

## **2. Important Definitions**
| **Term**                  | **Definition** |
|---------------------------|--------------|
| **Gradient Descent** | An optimization algorithm that updates model parameters by computing the gradient of a loss function. |
| **Learning Rate (\(\alpha\))** | Controls the step size when updating parameters in gradient descent. |
| **Stochastic Gradient Descent (SGD)** | Updates the model parameters using only one randomly selected training example at a time. |
| **Mini-batch Gradient Descent** | Uses a small batch of training examples for each gradient update, balancing efficiency and accuracy. |
| **Full-batch Gradient Descent** | Uses the entire dataset for each gradient update. |
| **Logistic Regression** | A classification model that applies the sigmoid function to predict probabilities. |
| **Softmax Function** | Converts raw scores into probabilities for multi-class classification: \( P(y=i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}} \). |
| **Cross-Entropy Loss** | Measures how well predicted probabilities match true labels, commonly used for classification tasks. |
| **Confidence Score** | The probability assigned by a classifier to a predicted class (e.g., 0.9 means 90% confidence). |

---

## **3. Solution Process in Formulas**

### **1. Classification with Confidence**
#### **Step 1: Logistic Regression Prediction**
\[
P(v=1 | x) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}
\]

#### **Step 2: Compute Cross-Entropy Loss**
\[
J(w) = -\frac{1}{N} \sum_{i=1}^{N} \left[ v_i \log(\sigma(w^T x_i)) + (1 - v_i) \log(1 - \sigma(w^T x_i)) \right]
\]

#### **Step 3: Compute Gradient for Logistic Regression**
\[
\frac{\partial J}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} (\sigma(w^T x_i) - v_i) x_i
\]
\[
\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\sigma(w^T x_i) - v_i)
\]

#### **Step 4: Update Rule for Logistic Regression**
\[
w^{(t+1)} = w^{(t)} - \alpha \frac{\partial J}{\partial w}
\]
\[
b^{(t+1)} = b^{(t)} - \alpha \frac{\partial J}{\partial b}
\]

#### **Step 5: Softmax Function for Multi-Class Classification**
For class \( i \), compute:
\[
P(y=i) = \frac{e^{w_i^T x}}{\sum_{j} e^{w_j^T x}}
\]

#### **Step 6: Compute Cross-Entropy Loss for Multi-Class**
\[
J(w) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} v_{ik} \log P(y=k | x_i)
\]

#### **Step 7: Update Rule for Multi-Class Logistic Regression**
\[
w^{(t+1)} = w^{(t)} - \alpha \frac{\partial J}{\partial w}
\]

---

## **4. Sample Numerical Example**
### **Example 1: Gradient Descent for Linear Regression**
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
\[
\frac{\partial J}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} (w x_i + b - v_i) x_i
\]
\[
\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (w x_i + b - v_i)
\]

#### **Step 5: Update Parameters**
\[
w^{(t+1)} = w^{(t)} - \alpha \frac{\partial J}{\partial w}
\]
\[
b^{(t+1)} = b^{(t)} - \alpha \frac{\partial J}{\partial b}
\]

---

## **5. Others (Additional Important Details)**
- **Why is Gradient Descent Needed?**  
  - Some functions do not have a simple formula for finding minima, so gradient descent helps optimize iteratively.

- **Learning Rate Choice:**  
  - Too **high**: Algorithm **overshoots**, fails to converge.
  - Too **low**: Converges **too slowly**.

- **Why Use Cross-Entropy Instead of MSE for Classification?**  
  - MSE gives poor gradients for classification problems, leading to slow learning.

- **Softmax vs. Sigmoid:**  
  - Sigmoid is for **binary classification**.
  - Softmax is for **multi-class classification**.