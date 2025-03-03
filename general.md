# **Structure Summary**

## **1. Machine Learning Categories**

| **Type**                | **Definition**  | **When to Use?**  | **Example**  |
|-------------------------|----------------|-------------------|--------------|
| **Supervised Learning (S)** | The dataset contains **both inputs (X) and outputs (Y)** (labeled data). The model learns a function \( f(X) \rightarrow Y \). | When the goal is to **predict an output based on known labeled data**. | Predicting house prices, spam detection, medical diagnosis. |
| **Unsupervised Learning (US)** | The dataset contains **only inputs (X)**, without labels. The model finds hidden structures in the data. | When the goal is to **group data into clusters or reduce dimensionality**. | Customer segmentation, anomaly detection, clustering similar news articles. |

---

## **2. Supervised Learning Breakdown**

| **Problem Type**          | **Goal**                                      | **Solution**                                       | **Loss Function** |
|---------------------------|----------------------------------------------|--------------------------------------------------|------------------|
| **Regression**            | Predict **continuous values** (e.g., price, temperature). | **Linear Regression** \( y = xw + b \)           | **Mean Squared Error (MSE)** |
| **Binary Classification** | Predict **two categories** (e.g., Spam vs. Not Spam). | **Logistic Regression** \( \sigma(xw + b) \)     | **Cross-Entropy Loss** |
| **Multi-Class Classification** | Predict **3+ categories** (e.g., Cat, Dog, Bird). | **Softmax Regression** (Generalized Logistic Regression) | **Cross-Entropy Loss** |
| **Finding a Decision Boundary** | Find the best **separation boundary** between two groups. | **Support Vector Machine (SVM)** | **Hinge Loss (Lagrange Optimization)** |

---

## **3. Unsupervised Learning Breakdown**

| **Problem Type**           | **Goal**                                      | **Solution** |
|----------------------------|----------------------------------------------|-------------|
| **Clustering**             | Group **similar** data points into clusters. | **K-Means Algorithm** (assigns each data point to one of \( K \) centroids). |
| **Dimensionality Reduction** | Reduce **data complexity** while preserving patterns. | **Principal Component Analysis (PCA)** (finds the best lower-dimensional representation of data). |

---

## **4. Core Algorithms & Their Details**

### **1. K-Means Clustering (Unsupervised)**
- **Goal:** Group data into **\( K \) clusters** without labels.
- **Steps:**
  1. Pick **\( K \) random centroids**.
  2. Assign **each point** to the closest centroid.
  3. **Recalculate centroids** as the mean of points in each cluster.
  4. Repeat until **centroids stop changing**.

---

### **2. Linear Regression (Supervised - Regression)**
- **Goal:** Find the best **straight-line fit** for continuous predictions.
- **Equation:**  
  \[
  y = xw + b
  \]
- **Optimization:** Uses **Mean Squared Error (MSE)**:
  \[
  J(w) = \frac{1}{N} \sum (y_i - w^T x_i)^2
  \]
- **Solution:**  
  - **Gradient Descent:** Iteratively updates \( w \) to minimize loss.
  - **Closed-Form Solution (Normal Equation):**  
    \[
    w^* = (X X^T)^{-1} X v
    \]

---

### **3. Logistic Regression (Supervised - Binary Classification)**
- **Goal:** Classify data into **two categories (0 or 1)** using probabilities.
- **Equation:**  
  \[
  \hat{y} = \sigma(xw + b) = \frac{1}{1 + e^{-(xw + b)}}
  \]
- **Loss Function:**  
  - Uses **Cross-Entropy Loss**, not MSE:
    \[
    J(w) = -\frac{1}{N} \sum \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
    \]
- **Optimization:**  
  - **Gradient Descent** (same as Linear Regression but with Cross-Entropy).

---

### **4. Softmax Regression (Supervised - Multi-Class Classification)**
- **Goal:** Classify data into **3+ categories (e.g., Dog, Cat, Bird)**.
- **Equation (Softmax Function):**  
  \[
  \hat{y}_i = \frac{e^{xw_i}}{\sum_{j} e^{xw_j}}
  \]
- **Loss Function:**  
  - **Cross-Entropy Loss for Multi-Class**:
    \[
    J(w) = -\frac{1}{N} \sum \sum_{k=1}^{K} y_k \log(\hat{y}_k)
    \]
- **Optimization:**  
  - Uses **Gradient Descent**.

---

### **5. Support Vector Machine (SVM) (Supervised - Decision Boundaries)**
- **Goal:** Find the **best hyperplane** that maximizes the margin between two classes.
- **Equation:**  
  \[
  w^T x + b = 0
  \]
- **Optimization Problem (Lagrange Method):**
  \[
  \min_{w, b} \frac{1}{2} ||w||^2
  \]
  **Subject to:**  
  \[
  y_n (w^T x_n + b) \geq 1
  \]
- **Why Use SVM?**
  - Maximizes margin between classes.
  - Works well for high-dimensional data.
  - Uses Kernel Trick for non-linear classification.

---

## **5. Key Concepts You Must Know**

| **Concept**                 | **Definition**  |
|-----------------------------|----------------|
| **Loss Function**            | Measures **how wrong** a model's prediction is (e.g., MSE for Regression, Cross-Entropy for Classification). |
| **Gradient Descent**         | An optimization method that **iteratively adjusts weights** to minimize the loss function. |
| **Bias-Variance Tradeoff**   | **Bias:** Model too simple (underfitting). **Variance:** Model too complex (overfitting). Need a balance. |
| **Regularization**           | Prevents overfitting by adding a penalty on large weights (L1 = Lasso, L2 = Ridge). |
| **Empirical Risk**           | The average loss **on training data**. |
| **True Risk**                | The expected loss on **unseen test data** (real-world performance). |
| **Cross-Validation**         | Splitting data into **training & validation** to test performance (e.g., K-Fold Cross-Validation). |

---

## **6. How to Choose the Right Algorithm**
- **Use K-Means if** → No labels, want to **group data**.
- **Use Linear Regression if** → Predicting **continuous values**.
- **Use Logistic Regression if** → Predicting **two categories (Binary Classification)**.
- **Use Softmax Regression if** → Predicting **multiple categories (Multi-Class Classification)**.
- **Use SVM if** → You need a **clear separation boundary** between two groups.