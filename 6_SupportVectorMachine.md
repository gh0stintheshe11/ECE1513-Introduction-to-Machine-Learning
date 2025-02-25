## Unit 6: Support Vector Machine (SVM)

### 1. Connecting to Previous Lectures

1. **From Linear Classification to SVM**  
   - We ended Lecture 5 with the idea of a **maximal-margin** boundary for binary classification.  
   - In **ideal** linearly separable scenarios, the “hard-margin SVC” sets up constraints: \(v_n(\mathbf{w}^\top\mathbf{x}_n)\geq 1\).  
   - Lecture 6 extends that to **non-separable** data (soft margin) and **nonlinear** data (via kernels).

2. **Why SVM?**  
   - SVMs can deliver robust decision boundaries, especially in high-dimensional or complex feature spaces, while controlling overfitting via margin.  
   - Offers a geometric notion of “confidence” (margin) as well as a powerful generalization (kernel trick).

---

### 2. Hard-Margin SVC: Review of the Basics

1. **Hard-Margin Constraints**  
   - For data \(\{(\mathbf{x}_n, v_n)\}\) with \(v_n\in\{-1,+1\}\), assume it’s linearly separable.  
   - We want \(\mathbf{w}\) such that \(v_n(\mathbf{w}^\top \mathbf{x}_n)\ge1\) for all \(n\).  
   - The **margin** is \(\tfrac{1}{\|\mathbf{w}\|}\). We **maximize** margin \(\Leftrightarrow\) **minimize** \(\|\mathbf{w}\|^2\).  
   - Primal problem:
     \[
       \min_{\mathbf{w}}\;\|\mathbf{w}\|^2
       \quad\text{subject to}\quad v_n(\mathbf{w}^\top\mathbf{x}_n)\ge1,\;\forall n.
     \]

2. **Geometry**  
   - If \(\mathbf{w}^\top\mathbf{x} = 0\) is the boundary, points satisfying \(\mathbf{w}^\top\mathbf{x}=1\) or \(\mathbf{w}^\top\mathbf{x}=-1\) are the **supporting hyperplanes**, and **support vectors** are the points lying exactly on those hyperplanes.

3. **Lagrange Dual**  
   - We form the Lagrangian \(\mathcal{L}(\mathbf{w}, \alpha)\) by introducing \(\alpha_n\ge0\) for each constraint:  
     \[
       \mathcal{L}(\mathbf{w},\boldsymbol{\alpha})
       =\|\mathbf{w}\|^2 - \sum_{n=1}^N \alpha_n\bigl[v_n\,(\mathbf{w}^\top\mathbf{x}_n)-1\bigr].
     \]
   - Stationary condition w.r.t. \(\mathbf{w}\) yields 
     \(\mathbf{w}=\tfrac12\sum_n \alpha_n\,v_n\,\mathbf{x}_n\).  
     (Sometimes a factor of \(\tfrac12\) appears depending on your notation for \(\|\mathbf{w}\|^2\).)

4. **Support Vectors**  
   - KKT’s complementary slackness states \(\alpha_n\,[\,v_n(\mathbf{w}^\top\mathbf{x}_n)-1]=0\).  
   - If \(v_n(\mathbf{w}^\top\mathbf{x}_n) >1\), then \(\alpha_n=0\).  
   - Only points with \(v_n(\mathbf{w}^\top\mathbf{x}_n)=1\) can have \(\alpha_n>0\).  
   - Those are the “support vectors” on the margin boundary.

---

### 3. Soft-Margin SVM: Handling Non-Separable Data

1. **Real-World Data**  
   - Usually, perfect separability is unrealistic. We allow some margin violations or misclassifications.  
   - Introduce **slack** \(\xi_n\ge0\) for each data point.

2. **Primal Form**  
   - \[
       \min_{\mathbf{w}, \{\xi_n\}} \quad \|\mathbf{w}\|^2 + C\sum_{n=1}^N \xi_n
       \quad \text{subject to}
       \quad v_n\,(\mathbf{w}^\top\mathbf{x}_n)\ge 1-\xi_n,\;\xi_n\ge0,
     \]
     - \(C>0\) is a hyperparameter controlling the trade-off.  
     - If \(\xi_n>1\), that point \(\mathbf{x}_n\) is **misclassified**.  
     - Minimizing \(\sum \xi_n\) tries to reduce the total margin violation.

3. **Dual Form & KKT**  
   - We get dual variables \(\alpha_n,\mu_n\ge0\). Now we have two sets of constraints: (1) margin constraints; (2) \(\xi_n\ge0\).  
   - The final solution for \(\mathbf{w}\) still sums over \(\alpha_n\,v_n\,\mathbf{x}_n\). But \(\alpha_n\le C\) typically arises from the new slack constraints.  
   - Points with \(\alpha_n=C\) are often significantly violating the margin or are misclassified.

4. **Interpretation**  
   - The margin is still \(\tfrac1{\|\mathbf{w}\|}\), but we pay a penalty \(C\sum_n \xi_n\).  
   - Adjusting \(C\) changes how “strict” vs. “loose” we are about margin violations.

---

### 4. Kernel Trick: Nonlinear SVM

1. **Motivation**  
   - Not all data is linearly separable in the original feature space. If we can map \(\mathbf{x}\to\phi(\mathbf{x})\) to a higher-dimensional or transformed space, maybe it becomes linearly separable there.  
   - Example: If data is in 2D but has a circle shape, we can map to \((x^2, y^2, \sqrt2\,xy)\) or something that is linearly separable in that new dimension.

2. **Kernel Method**  
   - The **dual** SVM solution only needs inner products \(\phi(\mathbf{x}_n)^\top\phi(\mathbf{x}_m)\).  
   - Define a **kernel** \(K(\mathbf{x}_n,\mathbf{x}_m)=\phi(\mathbf{x}_n)^\top \phi(\mathbf{x}_m)\).  
   - Then SVM classification is 
     \[
       \mathrm{sign}\Bigl[\sum_{n=1}^N \alpha_n\,v_n\,K(\mathbf{x}_n,\mathbf{x}) + b\Bigr].
     \]
   - We never explicitly compute \(\phi(\mathbf{x})\); we just compute \(K(\mathbf{x},\mathbf{z})\).  

3. **Popular Kernels**  
   - **Polynomial**: \(K(\mathbf{x},\mathbf{z})=(\mathbf{x}^\top\mathbf{z}+c)^p\).  
   - **Gaussian RBF**: \(K(\mathbf{x},\mathbf{z})=\exp\bigl(-\|\mathbf{x}-\mathbf{z}\|^2/(2\sigma^2)\bigr)\).  
   - **Sigmoid**: \(K(\mathbf{x},\mathbf{z})=\tanh(\alpha\,\mathbf{x}^\top\mathbf{z}+\beta)\).  
   - Must satisfy Mercer’s condition to be a valid kernel (positive semi-definite).

4. **Choosing & Tuning**  
   - **C** (slack penalty) and kernel parameters (\(\sigma\) for RBF, or degree \(p\) for polynomial) are typically found via **cross-validation** or a grid search.  
   - Overfitting can happen if the kernel is too flexible or \(C\) is too large.

---

### 5. Implementation & Practical Notes

1. **SMO Algorithm**  
   - A widely used approach to solve the SVM dual problem is **Sequential Minimal Optimization (SMO)**. It updates pairs of \(\alpha_n\) at a time.  
   - Complexity can become an issue for very large \(N\). Often consider approximate or linear SVM solvers.

2. **Prediction**  
   - Once trained, the decision function is 
     \[
       f(\mathbf{x})=\sum_{n\in SV}\alpha_n\,v_n\,K(\mathbf{x}_n,\mathbf{x})+b,
     \]
     where “\(SV\)” is the set of support vectors (those with \(\alpha_n>0\)).  
   - Typically \(\alpha_n=0\) for points not on or inside the margin, so large subsets of the dataset vanish at prediction time.

3. **Interpretation**  
   - Distance from the margin \(\mathbf{w}^\top \mathbf{x}+b\) is still a notion of “confidence.”  
   - For kernels, “margin” is in the new feature space \(\phi(\mathbf{x})\). But geometry still holds.

4. **Soft-Margin RBF SVM**  
   - Probably the most common SVM variant in practice:  
     - Slack variable approach for partial misclassification.  
     - Gaussian kernel \(K(\mathbf{x},\mathbf{z})=\exp(-\|\mathbf{x}-\mathbf{z}\|^2/\,(2\sigma^2))\).  
     - Hyperparameters: \(C\) and \(\sigma\). You tune them to find a good balance between margin, errors, and how “curvy” the boundary can be.

---

### 6. Common Pitfalls & Final Observations

1. **Scaling**  
   - For SVM, especially with kernels, it’s crucial to **scale** or **normalize** features (e.g., zero mean, unit variance) so that no dimension dominates the distance measure in the RBF kernel, etc.

2. **Kernel Choice**  
   - RBF is often a safe default. Polynomial kernels can be useful if you suspect polynomial-like data. Sigmoid kernel is less common outside certain contexts.

3. **Comparisons**  
   - **SVM** vs. **Logistic Regression**:  
     - SVM → margin-based approach, no direct probability interpretation (though you can get calibrations or Platt scaling).  
     - Logistic → direct probability output \(\sigma(\cdot)\), cross-entropy optimization.  
   - **SVM** vs. **Neural Nets**: Historically, SVM was top for many classification tasks before deep learning advanced. SVM can still excel in moderate data settings or with specialized kernels.

4. **Regularization & Overfitting**  
   - \(C\) and kernel parameters help control overfitting. Large \(C\) → tries to fit training data “too well.” For RBF, small \(\sigma\) can lead to extremely wiggly boundaries. Cross-validation is essential.

5. **Sparsity**  
   - One appealing feature of SVM: typically only a fraction of data points become support vectors. Others have \(\alpha_n=0\). That can help with inference speed in moderate datasets.

---

### 7. Summary of Key Equations

1. **Soft-Margin Primal**:
   \[
     \min_{\mathbf{w},\,\boldsymbol{\xi}} \|\mathbf{w}\|^2 + C\sum_{n=1}^N \xi_n,
     \quad
     \text{subject to }v_n\,(\mathbf{w}^\top\mathbf{x}_n)\ge1-\xi_n,\;\xi_n\ge0.
   \]

2. **Soft-Margin Dual** (schematic):
   \[
     \begin{aligned}
       &\max_{\boldsymbol{\alpha}} \quad \sum_{n=1}^N \alpha_n - \tfrac12\sum_{n,m}\alpha_n \,\alpha_m\,v_n\,v_m\,(\mathbf{x}_n^\top\mathbf{x}_m)
       \\
       &\text{subject to } 0\le\alpha_n\le C,\;\;\sum_{n=1}^N \alpha_n\,v_n=0.
     \end{aligned}
   \]
   (In practice, details differ slightly if we incorporate slacks into the Lagrangian carefully.)

3. **Kernel Trick**:
   \[
     \mathbf{x}_n^\top \mathbf{x}_m \;\to\; K(\mathbf{x}_n,\mathbf{x}_m)
     \;=\;\phi(\mathbf{x}_n)^\top\phi(\mathbf{x}_m).
   \]

4. **Decision Function**:
   \[
     f(\mathbf{x})
     = \mathrm{sign}\Bigl(\sum_{n\in SV}\alpha_n\,v_n\,K(\mathbf{x}_n,\mathbf{x})+b\Bigr).
   \]