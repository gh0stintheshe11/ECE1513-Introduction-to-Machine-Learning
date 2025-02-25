## Unit 4: Linear Regression & Classification

### 1. Transition to Supervised Learning

1. **Big Picture**  
   - We completed **Unsupervised** methods: Clustering, Distribution Learning, PCA.  
   - **Now** we shift to **Supervised Learning** (with **labeled** data \(\{(\mathbf{x}_n, v_n)\}\)).  
   - Topics in **Lecture 4**:  
     - Linear **Regression** (continuous labels)  
     - Linear **Classification** (discrete labels), focusing on simple threshold-based or linear discriminant

2. **General Supervised Recipe**  
   - **Data**: \(\{(\mathbf{x}_n, v_n)\}_{n=1}^N\).  
   - **Model**: \(f:\mathbf{x}\mapsto y\) belongs to a hypothesis set \(\mathcal{H}\) (e.g., linear or polynomial).  
   - **Learning Algorithm**: Minimizes an **empirical risk** \(\hat{R}(f)\) measured by a **loss function** \(L(y,v)\).

---

### 2. Linear Regression

1. **Setup**  
   - Suppose \(v_n\in\mathbb{R}\) is a **continuous** target (like price, height, temperature).  
   - A **linear model**: 
     \[
       f(\mathbf{x})=\mathbf{w}^\top\mathbf{x} + b\quad\text{(or augment }\mathbf{x}\text{ with 1 to absorb }b\text{).}
     \]

2. **Polynomial Regression**  
   - A generalization: let \(\phi(\mathbf{x})\) be polynomial features (or any feature mapping). Then 
     \[
       f(\mathbf{x})=\mathbf{w}^\top\phi(\mathbf{x}).
     \]
   - E.g. in 1D, we might use \(\phi(x)=[1, x, x^2, \dots, x^P]\) to fit a polynomial of degree \(P\).

3. **Loss Function & Empirical Risk**  
   - Commonly use **squared loss**: \(L(y,v)=\tfrac{1}{2}(y-v)^2\). (The \(\tfrac12\) factor is optional, can simplify derivatives.)  
   - Empirical risk:
     \[
       \hat{R}(\mathbf{w})=\frac{1}{N}\sum_{n=1}^N \bigl[\mathbf{w}^\top\mathbf{x}_n + b - v_n\bigr]^2.
     \]

4. **Closed-Form Solution**  
   - In matrix form, with \(\mathbf{X}\in\mathbb{R}^{d\times N}\) (columns are \(\mathbf{x}_n\)) and \(\mathbf{v}\in\mathbb{R}^N\):
     \[
       \hat{R}(\mathbf{w})=\frac{1}{N}\|\mathbf{X}^\top \mathbf{w} - \mathbf{v}\|^2.
     \]
   - Taking gradient \(\nabla_{\mathbf{w}}\hat{R}(\mathbf{w})=0\) leads to **normal equations**:
     \[
       \mathbf{X}\mathbf{X}^\top \mathbf{w} = \mathbf{X}\mathbf{v}.
     \]
   - **If** \(\mathbf{X}\mathbf{X}^\top\) is invertible (or we use the pseudo-inverse \(\mathbf{X}^+\)), we get:
     \[
       \mathbf{w}^*=\bigl(\mathbf{X}\mathbf{X}^\top\bigr)^{-1}\mathbf{X}\,\mathbf{v}.
     \]

5. **Convexity & Uniqueness**  
   - Squared loss is **convex** in \(\mathbf{w}\) → the above solution is a **global minimum**.  
   - If \(\mathbf{X}\mathbf{X}^\top\) is singular (rank-deficient), many solutions can exist or we need regularization.

6. **Examples**  
   - **Polynomial** fit on \(\{(x_n,v_n)\}\). The slides might show:  
     - You can choose a polynomial degree \(P\), transform each \(x_n\) → \(\phi(x_n)\), solve linear regression in that expanded space.  
   - **Interpretation**: Minimizing squared difference from data points.

---

### 3. Linear Classification

1. **Setup**  
   - We want to predict a **discrete** label (e.g. 0 or 1, or \(\{-1,+1\}\)).  
   - A **linear classifier**:  
     \[
       y=\mathrm{sign}\bigl(\mathbf{w}^\top \mathbf{x} + b\bigr)\quad \text{(binary classification)}.
     \]

2. **Threshold / Perceptron Approach**  
   - **0–1 Loss**: \(\hat{R}(\mathbf{w})=\tfrac1N \sum_{n=1}^N \mathbf{1}\{y_n \neq v_n\}\).  
   - Minimizing 0–1 loss is generally **hard** (non-differentiable, often NP-hard).  
   - The **Perceptron Algorithm** is a classical iterative update that tries to find \(\mathbf{w}\) if data are linearly separable.

3. **Regression Trick**  
   - One naive approach is treat label \(\{-1,+1\}\) and do a “regression” fit \(\mathbf{w}^\top \mathbf{x}\approx v\).  
   - But can be poor if outliers or if data not easily matched by a linear function with small squared error.

4. **Logistic (Preview)**  
   - We can define \(p=\sigma(\mathbf{w}^\top\mathbf{x}+b)\) to be the probability of class=1, with \(\sigma(\cdot)\) the sigmoid.  
   - Minimizing **cross-entropy** then leads to a better classification. (Lecture 5 typically covers logistic regression in detail.)

---

### 4. Risk Minimization & Examples in the Slides

1. **Risk = Expected Loss**  
   - For regression: often use squared loss.  
   - For classification: we might use 0–1 loss, or logistic/cross-entropy loss.  
2. **Examples**  
   - The lecture might show a numeric example: weight vs. pet type (cat/dog). A simple threshold or line in 1D or 2D.  
   - Another example: polynomial curve fitting with regression on a small data set, demonstrating underfitting vs. overfitting.

3. **Bias vs. Variance** (Possibly introduced)  
   - Some slides mention that high-degree polynomials can overfit, while a linear function might underfit.  
   - This sets the stage for **regularization** and **model complexity** topics in advanced lectures.

4. **Gradient-Based Methods**  
   - Even though linear regression has a closed-form, we can also solve it via gradient descent.  
   - For classification with logistic or 0–1 loss, we *must* use iterative methods (no closed-form).

---

### 5. Practical Takeaways

1. **Linear Models**: Quick, relatively simple approach for both regression and classification tasks.  
2. **Closed-Form vs. Iterative**:  
   - **Linear Regression**: normal equations if feasible; or gradient-based for large data.  
   - **Classification**: no closed-form for 0–1 loss. We rely on perceptron or logistic regression.  
3. **Hyperparameters**: polynomial degree \(P\), or how we handle the bias term, scaling, etc.  
4. **Generalization**: check train vs. validation/test performance to detect over/underfitting.