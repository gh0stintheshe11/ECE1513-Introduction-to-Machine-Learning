## **Unit 6: Support Vector Machines (SVM)**  

---

### **1. General Idea**  
- **Problem We Are Solving**:  
  - We want to classify data into two categories by finding the **optimal decision boundary** (separating hyperplane).  
  - Unlike logistic regression, SVM **maximizes the margin** between the decision boundary and the closest data points (support vectors).  

- **Model We Use**:  
  - **Hard-margin SVM**: Assumes perfectly separable data, maximizes the margin without allowing misclassification.  
  - **Soft-margin SVM**: Allows misclassification with a penalty term to handle overlapping data.  
  - **Kernelized SVM**: Maps data to higher-dimensional space for cases where data is not linearly separable.  

---

### **2. Important Definitions**  

| **Term**                  | **Definition** |
|---------------------------|--------------|
| **Hyperplane** | The decision boundary that separates different classes in SVM. |
| **Margin** | The distance between the hyperplane and the closest support vectors. |
| **Support Vectors** | Data points that are closest to the hyperplane and determine its position. |
| **Hard-margin SVM** | Assumes perfectly separable data and maximizes the margin without allowing misclassification. |
| **Soft-margin SVM** | Introduces a slack variable \( \xi \) to allow some misclassification for better generalization. |
| **Slack Variable (\(\xi\))** | A variable that allows some misclassified points to exist in soft-margin SVM. |
| **Hinge Loss** | The loss function used in SVM to ensure correct classification while maximizing the margin. |
| **Kernel Trick** | A technique used to transform data into a higher-dimensional space where it becomes linearly separable. |
| **Lagrangian Multipliers (\(\alpha_i\))** | Variables used to solve the optimization problem in SVM. |

---

### **3. Solution Process in Formulas**  

#### **Step 1: Define the Hyperplane**  
For an input \( x \), the decision function is:  
\[
f(x) = w^T x + b
\]
where \( w \) is the weight vector and \( b \) is the bias.  

#### **Step 2: Define the Classification Condition**  
For correct classification:
\[
y_i (w^T x_i + b) \geq 1, \quad \forall i
\]
where \( y_i \in \{-1, 1\} \) represents class labels.

#### **Step 3: Define the Optimization Problem**  
We want to **maximize the margin** \( \frac{1}{\| w \|} \), which is equivalent to minimizing:  
\[
\frac{1}{2} \|w\|^2
\]

Subject to the constraint:  
\[
y_i (w^T x_i + b) \geq 1, \quad \forall i
\]

#### **Step 4: Introduce Lagrangian and Solve**  
We introduce Lagrange multipliers \( \alpha_i \) and define:  
\[
L(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i} \alpha_i [y_i (w^T x_i + b) - 1]
\]

Taking derivatives and solving gives:  
\[
w = \sum_{i} \alpha_i y_i x_i
\]

Solving for \( \alpha \), we get:  
\[
\max_{\alpha} \sum_{i} \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j
\]

Subject to:  
\[
\sum_{i} \alpha_i y_i = 0, \quad \alpha_i \geq 0
\]

#### **Step 5: Soft-Margin SVM (Allowing Misclassification)**  
For soft-margin SVM, we introduce **slack variables** \( \xi_i \):  
\[
y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
\]

The objective function becomes:  
\[
\frac{1}{2} \|w\|^2 + C \sum_{i} \xi_i
\]

where \( C \) controls the trade-off between maximizing the margin and allowing misclassification.

#### **Step 6: Kernel Trick for Non-Linearly Separable Data**  
If data is **not linearly separable**, apply the **kernel trick** \( K(x_i, x_j) \):  
\[
\max_{\alpha} \sum_{i} \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)
\]

where common kernels are:  
- **Linear Kernel**: \( K(x, y) = x^T y \)  
- **Polynomial Kernel**: \( K(x, y) = (x^T y + c)^d \)  
- **RBF (Gaussian) Kernel**: \( K(x, y) = \exp(-\gamma \| x - y \|^2) \)  

---

### **4. Sample Numerical Example**
#### **Example 1: Compute the Optimal Hyperplane**
**Given Data:**
| \( x_1 \) | \( x_2 \) | \( y \) |
|--------|--------|--------|
| 2      | 2      | +1     |
| 1      | -1     | -1     |
| -2     | -2     | -1     |

**Step 1: Define the Hyperplane Equation**  
\[
w^T x + b = 0
\]

**Step 2: Compute Decision Boundary**
For support vectors satisfying \( y_i (w^T x_i + b) = 1 \):

Using the first support vector \( (2,2) \):
\[
w_1(2) + w_2(2) + b = 1
\]

Using the second support vector \( (-2,-2) \):
\[
w_1(-2) + w_2(-2) + b = -1
\]

Solving these equations, we obtain:  
\[
w_1 = 0.5, \quad w_2 = 0.5, \quad b = -1
\]

Final **decision boundary**:
\[
0.5 x_1 + 0.5 x_2 - 1 = 0
\]

---

### **5. Others (Additional Important Details)**
- **Why Maximize the Margin?**  
  - A **larger margin** means **better generalization** to unseen data.  
  - Unlike logistic regression, which optimizes classification probabilities, SVM focuses on **separating classes robustly**.  

- **How Do Slack Variables Help?**  
  - They allow **some points** to be inside the margin, improving SVM’s ability to **handle noisy data**.  

- **Kernelized SVM and Feature Space Transformation**  
  - If data **is not linearly separable**, **kernel functions** map data into a **higher-dimensional space** where it becomes linearly separable.  