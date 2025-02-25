# Unit 3: Principle Component Analysis (PCA) & Dimensionality Reduction

### 1. Motivation & Key Concepts

1. **Dimensionality Reduction**  
   - High-dimensional data (potentially thousands of features) can be *difficult* to visualize or model.  
   - **Goal**: Represent or compress data into fewer dimensions (say, \(K < D\)) while preserving as much “signal” as possible.  
   - **Applications**: Data compression, visualization, noise reduction, and feature extraction.

2. **Why PCA?**  
   - PCA is a **linear** approach to dimensionality reduction.  
   - It projects data from \(D\) dimensions down to \(K\) dimensions in a way that (1) **maximizes the variance** of the projected data or (2) **minimizes reconstruction error**.

3. **Example Motivations**  
   - **Image Compression**: Each image is a high-dimensional vector (e.g., \(300\times200 = 60000\) pixels). We want to store it with fewer numbers.  
   - **Recommendation Systems**: Large user–item rating matrices often lie on a lower-dimensional “latent” space of preferences.

---

### 2. Data Representation & Linear Algebra Review

1. **Representing Data**  
   - Dataset \(D = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}\), each \(\mathbf{x}_n \in \mathbb{R}^D\).  
   - We want a projection \(\mathbf{z}_n \in \mathbb{R}^K\), with \(K < D\).

2. **Orthonormal Bases**  
   - A set of vectors \(\{\mathbf{u}_1, \dots, \mathbf{u}_K\}\) in \(\mathbb{R}^D\) is orthonormal if \(\mathbf{u}_i^\top \mathbf{u}_j = 0\) (for \(i\neq j\)) and \(\|\mathbf{u}_i\|=1\).  
   - We often collect them into a matrix \(\mathbf{U}\in\mathbb{R}^{D\times K}\) with \(\mathbf{U}^\top \mathbf{U} = \mathbf{I}_K\).

3. **Eigenvalues & Covariance**  
   - For a symmetric matrix \(\mathbf{A}\in\mathbb{R}^{D\times D}\), an eigenvector \(\mathbf{v}\) satisfies \(\mathbf{A}\,\mathbf{v} = \lambda \mathbf{v}\).  
   - **Sample Covariance** \(\mathbf{\Sigma}\) is computed from mean-centered data:  
     \[
       \mathbf{\Sigma} = \tfrac{1}{N}\sum_{n=1}^N (\mathbf{x}_n - \mathbf{\mu})\,(\mathbf{x}_n - \mathbf{\mu})^\top,
     \]
     where \(\mathbf{\mu}=\tfrac{1}{N}\sum_{n=1}^N \mathbf{x}_n\).

---

### 3. Principle Component Analysis (PCA)

1. **Problem Formulation**  
   - We look for \(K\) directions (principal components) \(\{\mathbf{u}_1, \dots, \mathbf{u}_K\}\) onto which data will be projected.  
   - For each point \(\mathbf{x}_n\), the low-dimensional representation is  
     \[
       \mathbf{z}_n = \begin{bmatrix}
         \mathbf{u}_1^\top(\mathbf{x}_n - \mathbf{\mu}) \\
         \vdots \\
         \mathbf{u}_K^\top(\mathbf{x}_n - \mathbf{\mu})
       \end{bmatrix}.
     \]
   - We can optionally reconstruct:  
     \[
       \hat{\mathbf{x}}_n = \mathbf{\mu} + \sum_{k=1}^K z_{n,k}\,\mathbf{u}_k.
     \]

2. **Minimizing Reconstruction Error**  
   - The average reconstruction error is  
     \[
       \tfrac{1}{N}\sum_{n=1}^N \|\hat{\mathbf{x}}_n - \mathbf{x}_n\|^2.
     \]
   - PCA chooses \(\{\mathbf{u}_1, \dots, \mathbf{u}_K\}\) to minimize this error.

3. **Maximizing Variance View**  
   - Equivalently, PCA picks the subspace spanned by the top-\(K\) eigenvectors of \(\mathbf{\Sigma}\) (the largest \(K\) eigenvalues).  
   - The first principal component \(\mathbf{u}_1\) is the eigenvector of \(\mathbf{\Sigma}\) with the largest eigenvalue \(\lambda_1\).  
   - Next principal components \(\mathbf{u}_2, \dots, \mathbf{u}_K\) are eigenvectors for \(\lambda_2, \dots, \lambda_K\), each orthogonal to the previous ones.

4. **Algorithm Steps (High-Level)**  
   - **(a)** Compute mean \(\mathbf{\mu} = \tfrac{1}{N}\sum_{n=1}^N \mathbf{x}_n\).  
   - **(b)** Form the sample covariance \(\mathbf{\Sigma}\).  
   - **(c)** Find the top \(K\) eigenvectors (largest eigenvalues) of \(\mathbf{\Sigma}\).  
   - **(d)** Construct projection matrix \(\mathbf{U} = [\,\mathbf{u}_1, \dots, \mathbf{u}_K\,]\).  
   - **(e)** For each \(\mathbf{x}_n\), compute \(\mathbf{z}_n = \mathbf{U}^\top(\mathbf{x}_n - \mathbf{\mu})\).  
   - **(f)** Optional reconstruction: \(\hat{\mathbf{x}}_n = \mathbf{U}\,\mathbf{z}_n + \mathbf{\mu}\).

---

### 4. Simple Examples & Applications

1. **Line Example in 2D**  
   - If \(\{\mathbf{x}_n\}\) all lie on a single line, then you only need 1 principal component to represent them with zero error. The smaller eigenvalue will be \(0\).

2. **Image Compression**  
   - Each image \(\mathbf{x}_n\) is a large vector in \(\mathbb{R}^D\).  
   - PCA finds a rank-\(K\) approximation, letting you store the top PCA basis (“eigen-images”) plus coordinates \(\mathbf{z}_n\).  
   - Reconstruction quality improves with \(K\) but so does storage size.

3. **Matrix Completion**  
   - A large user–item rating matrix can be treated like data in \(\mathbb{R}^D\).  
   - PCA (or SVD) captures it in a lower-dimensional latent space if it’s approximately low rank.

---

### 5. Practical Considerations

1. **Data Preprocessing**  
   - Always **center** data by subtracting \(\mathbf{\mu}\).  
   - Optionally, **scale** each dimension (normalization) so no dimension unfairly dominates variance.

2. **Cost & Complexity**  
   - Naively, eigen-decomposition of a \(D\times D\) covariance is \(O(D^3)\). This is feasible for moderate \(D\).  
   - For very large \(D\), use incremental or randomized PCA.

3. **PCA is Linear**  
   - If data lie on a non-linear manifold, PCA may not capture it well.  
   - Solutions: **Kernel PCA**, **Autoencoders**, or other nonlinear approaches.

4. **Eigenvalue “Elbow”**  
   - Plot eigenvalues \(\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D\).  
   - Choose \(K\) where the curve “flattens out,” balancing dimension reduction vs. info loss.

5. **Sensitivity to Outliers**  
   - Large outliers can skew the principal components significantly. Possibly use **robust PCA** methods if outliers are frequent.

---

### 6. Study Tips & Recap

- **Understand** the two main derivations of PCA:  
  1. **Minimize** reconstruction error  
  2. **Maximize** variance of projected data  
- **Practice** computing covariance for small examples (2D or 3D) and doing manual eigen-decomposition.  
- **Remember** the final step: PCA is just \(\mathbf{U}^\top(\mathbf{x}-\mathbf{\mu})\) for the top eigenvectors \(\mathbf{u}_k\).  
- **Hyperparameter** \(K\) must be chosen, often by looking at how much variance we keep.