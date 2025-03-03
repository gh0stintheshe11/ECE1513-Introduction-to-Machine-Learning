# **Unit 3: Principal Component Analysis (PCA)**

---

## **1. General Idea**
### **What problem are we solving?**
- **Goal:** Reduce the dimensionality of data while preserving as much information (variance) as possible.
- **Key Idea:** We transform the dataset into a new coordinate system where the most significant features (Principal Components) capture the majority of the variance.
- **Approach:**  
  - We compute the **covariance matrix** of the data.
  - We find its **eigenvalues and eigenvectors**.
  - We project the data onto the **top $K$ eigenvectors** that explain the most variance.

### **What models are used?**
1. **Principal Component Analysis (PCA)**:  
   - A **linear transformation** that finds the **directions of maximum variance** in high-dimensional data.
   - Each **Principal Component (PC)** is an **eigenvector** of the **covariance matrix**.
  
2. **Covariance Matrix**:  
   - Measures how different features **vary together**.
   - Used to compute eigenvalues and eigenvectors for PCA.

---

## **2. Definitions**
These are the key terms and definitions you must know.

| **Term**                | **Definition** |
|------------------------|--------------|
| **Dimensionality Reduction** | The process of reducing the number of features while keeping important information. |
| **Principal Component Analysis (PCA)** | A transformation that projects data onto directions (eigenvectors) that maximize variance. |
| **Covariance Matrix $\Sigma$** | A matrix that shows the relationships between different variables: $\Sigma = \frac{1}{N} X^T X$. |
| **Eigenvalues $\lambda$** | Scalars that measure the variance captured by each eigenvector. |
| **Eigenvectors $v$** | Directions along which data has the most variance. |
| **Projection Matrix $U$** | The matrix containing the **top $K$ eigenvectors**, used for reducing dimensions. |
| **Reconstruction Error** | The difference between the original data and its projection. |
| **Latent Representation** | The new low-dimensional representation of the data after applying PCA. |

---

## **3. Solution Process**
This section provides **clear, step-by-step** derivations for **PCA algorithm and reconstruction**.

### **Step 1: Compute the Covariance Matrix**
1. **Center the dataset** (subtract the mean):
   $$
   \bar{x} = \frac{1}{N} \sum x_n, \quad X' = X - \bar{x}
   $$
2. **Compute the covariance matrix**:
   $$
   \Sigma = \frac{1}{N} X'^T X'
   $$

---

### **Step 2: Compute Eigenvalues and Eigenvectors**
- Solve for the **eigenvalues** $\lambda$ and **eigenvectors** $v$ of the covariance matrix:
  $$
  \Sigma v = \lambda v
  $$
- The eigenvectors **define new axes** for the transformed data.

---

### **Step 3: Select the Top $K$ Eigenvectors**
- **Sort eigenvalues in descending order**:  
  - The **largest eigenvalues correspond to the most significant principal components**.
- **Choose the top $K$ eigenvectors** to form the projection matrix:
  $$
  U = [v_1, v_2, ..., v_K]
  $$

---

### **Step 4: Project Data onto New Basis**
- Compute the **new low-dimensional representation**:
  $$
  Z = U^T X'
  $$
- The dataset is now represented in a **lower-dimensional space**.

---

### **Step 5: Reconstruct Data from PCA**
- If we want to **recover an approximation** of the original data:
  $$
  \hat{X} = U Z + \bar{x}
  $$
- The difference between $X$ and $\hat{X}$ is the **reconstruction error**.

---

## **4. Sample Numerical Example**
### **Problem Statement:**
We have the dataset:
$$
D = \{ (1, 2), (2, 4), (3, 6), (4, 8) \}
$$
We want to apply **PCA** to reduce it to **1 dimension**.

### **Step 1: Compute Mean and Center the Data**
$$
\bar{x} = \frac{1+2+3+4}{4} = 2.5, \quad \bar{y} = \frac{2+4+6+8}{4} = 5
$$

$$
X' = \begin{bmatrix}
-1.5 & -3 \\
-0.5 & -1 \\
0.5 & 1 \\
1.5 & 3
\end{bmatrix}
$$

---

### **Step 2: Compute Covariance Matrix**
$$
\Sigma = \frac{1}{4} X'^T X'
$$

$$
= \frac{1}{4} \begin{bmatrix}
(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2 & (-1.5)(-3) + (-0.5)(-1) + (0.5)(1) + (1.5)(3) \\
(-3)(-1.5) + (-1)(-0.5) + (1)(0.5) + (3)(1.5) & (-3)^2 + (-1)^2 + (1)^2 + (3)^2
\end{bmatrix}
$$

$$
= \frac{1}{4} \begin{bmatrix} 
2.5 & 5 \\ 
5 & 10 
\end{bmatrix}
$$

$$
= \begin{bmatrix} 
0.625 & 1.25 \\ 
1.25 & 2.5 
\end{bmatrix}
$$

---

### **Step 3: Compute Eigenvalues and Eigenvectors**
- **Solve $\text{det}(\Sigma - \lambda I) = 0$**:
  $$
  \begin{vmatrix} 0.625 - \lambda & 1.25 \\ 1.25 & 2.5 - \lambda \end{vmatrix} = 0
  $$

- **Eigenvalues**: $\lambda_1 = 3.125, \lambda_2 = 0$
- **Eigenvectors**: $v_1 = [0.4, 0.8]^T$, $v_2 = [-0.8, 0.4]^T$

---

### **Step 4: Project Data onto 1D**
Using **top eigenvector $v_1 = [0.4, 0.8]^T$**:
$$
z_n = v_1^T X'
$$

$$
Z = \begin{bmatrix} 
-1.5 & -3 \\ 
-0.5 & -1 \\ 
0.5 & 1 \\ 
1.5 & 3 
\end{bmatrix}
\begin{bmatrix} 
0.4 \\ 
0.8 
\end{bmatrix}
$$

$$
= \begin{bmatrix} 
-3.6 \\ 
-1.2 \\ 
1.2 \\ 
3.6 
\end{bmatrix}
$$

✔ **Dataset is now 1D: $Z = [-3.6, -1.2, 1.2, 3.6]$**.

---

## **5. Other Important Details**
- **PCA is sensitive to scaling**: Always **normalize features** before applying PCA.
- **PCA is linear**: It only works if data follows **linear patterns**; for non-linear data, use **Kernel PCA** or **Autoencoders**.
- **Eigenvalues represent variance**: **Larger eigenvalues capture more variance**.