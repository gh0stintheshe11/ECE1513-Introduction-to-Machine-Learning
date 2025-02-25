# Unit 5: Gradient Descent & Classification with Confidence

### 1. Maximum Likelihood Classification & Cross-Entropy

1. **Why Maximum Likelihood?**  
   - In a **supervised** setting with labels \(v_n \in \{0,1\}\) (binary classification), we can model  
     \[
       \mathbb{P}(v_n=1 \mid \mathbf{x}_n;\,\mathbf{w}) \;=\; \sigma(\mathbf{w}^\top \mathbf{x}_n),
     \]
     where \(\sigma(z)=\frac{1}{1 + e^{-z}}\) is the **sigmoid** function.  
   - Interpreting \(\sigma(\mathbf{w}^\top \mathbf{x}_n)\) as the **probability** of class 1 is the core assumption in **logistic regression**.  

2. **Dataset Likelihood**  
   - For an i.i.d. dataset \(D=\{\mathbf{x}_n, v_n\}_{n=1}^N\), the likelihood is
     \[
       L(\mathbf{w}) 
       = \prod_{n=1}^N \Bigl[\sigma(\mathbf{w}^\top \mathbf{x}_n)\Bigr]^{v_n}\,\Bigl[\,1-\sigma(\mathbf{w}^\top \mathbf{x}_n)\Bigr]^{1-v_n}.
     \]
   - Taking **log-likelihood** (\(\ln L(\mathbf{w})\)) turns products into sums.

3. **Cross-Entropy Loss**  
   - **Maximizing** the log-likelihood is equivalent to **minimizing** the **negative** log-likelihood:  
     \[
       \ell(\mathbf{w})
       = -\frac{1}{N}\sum_{n=1}^N \Bigl[v_n\ln \sigma(\mathbf{w}^\top \mathbf{x}_n)\;+\;(1-v_n)\ln\bigl(1-\sigma(\mathbf{w}^\top \mathbf{x}_n)\bigr)\Bigr].
     \]
   - This objective is known as the **cross-entropy** loss in binary classification.  
   - Cross-entropy is **differentiable** and **convex** in \(\mathbf{w}\), allowing gradient-based optimization.

4. **Why Not 0–1 Loss?**  
   - 0–1 loss \(\mathbf{1}\{f(\mathbf{x}_n)\neq v_n\}\) is **non-differentiable**; direct minimization is typically hard.  
   - Cross-entropy is a smooth surrogate that not only helps find a good classifier but also yields **probability estimates** (confidence).

5. **Gradient of Cross-Entropy**  
   - For logistic regression, the partial derivative often appears as:
     \[
       \nabla_{\mathbf{w}} \ell(\mathbf{w})
       = \frac{1}{N}\sum_{n=1}^N \Bigl[\sigma(\mathbf{w}^\top \mathbf{x}_n) - v_n\Bigr]\;\mathbf{x}_n.
     \]
   - We can then update \(\mathbf{w}\) via gradient descent methods.

---

### 2. Gradient Descent (Expanded)

1. **Motivation**  
   - Many ML objectives (e.g. cross-entropy for classification, or more complex neural-network losses) **lack** a closed-form solution.  
   - We use **gradient descent** to iteratively refine parameters \(\mathbf{w}\).

2. **General Algorithm**  
   - We want to **minimize** \(R(\mathbf{w})\). The update rule is:
     \[
       \mathbf{w}^{(t+1)}
       \;=\;\mathbf{w}^{(t)} \;-\; \eta\,\nabla R\bigl(\mathbf{w}^{(t)}\bigr),
     \]
     where \(\eta\) is the **learning rate**.  
   - Stop when changes are small or a maximum iteration/time is reached.

3. **Variants**  
   - **Batch Gradient Descent**: Uses the full dataset for each gradient step.  
   - **Stochastic Gradient Descent (SGD)**: Uses one sample (or a small mini-batch) at a time—faster on large datasets but noisier updates.  
   - **Mini-Batch GD**: A compromise, using small batches, is common in practice.

4. **Convergence & Learning Rate**  
   - If \(R\) is **convex** (e.g., linear or logistic regression), gradient descent converges to the **global** optimum for a suitably chosen \(\eta\).  
   - **Too large \(\eta\)** → might diverge or oscillate; **too small** \(\eta\) → slow training.  
   - Many heuristics (e.g. line search, schedules, or adaptive optimizers like Adam) can help.

5. **Example**: Gradient Descent for Logistic Regression  
   - We plug in the cross-entropy gradient:
     \[
       \mathbf{w}\;\leftarrow\;\mathbf{w}\;-\;\eta\,\frac{1}{N}\sum_{n=1}^N \bigl[\sigma(\mathbf{w}^\top \mathbf{x}_n)-v_n\bigr]\,\mathbf{x}_n.
     \]
   - Repeat until convergence.

---

### 3. Classification with Confidence

1. **Confidence in Decisions**  
   - In **logistic regression**, \(\sigma(\mathbf{w}^\top\mathbf{x})\) is the model’s probability that \(\mathbf{x}\) belongs to class 1. That **confidence** can guide threshold or further decisions.  
   - Example: If \(\sigma(\mathbf{w}^\top \mathbf{x})=0.9\), we’re quite confident the label is class 1.

2. **Maximal Margin Perspective**  
   - An alternative notion of “confidence” comes from how **far** a data point is from the decision boundary.  
   - This leads to **Support Vector Classifiers** (SVC) or **Support Vector Machines** (SVM).

3. **Support Vector Classifier** (Informal Intro)  
   - Suppose data are linearly separable. We want a classifier \(y=\mathrm{sign}(\mathbf{w}^\top \mathbf{x})\) that:  
     - **No training error**: \(v_n(\mathbf{w}^\top \mathbf{x}_n)>0\)\(\forall n\).  
     - **Large margin**: The margin is \(\frac{1}{\|\mathbf{w}\|}\). Maximizing margin \(\Leftrightarrow\) minimizing \(\|\mathbf{w}\|^2\).  
   - The **hard-margin SVC** problem:
     \[
       \min_{\mathbf{w}}\;\|\mathbf{w}\|^2
       \quad\text{subject to}\;v_n\,\mathbf{w}^\top \mathbf{x}_n\;\ge\;1,\;\forall n.
     \]
   - A “large margin” means points are not just correctly classified but also kept at a comfortable distance from the boundary → higher confidence.

4. **Why Margin = Confidence?**  
   - Points far from the boundary are less “risky” to misclassify if small perturbations happen.  
   - This geometry-based approach differs from logistic regression’s probability-based approach—both yield linear boundaries but interpret “confidence” differently.

5. **SVC vs. Logistic** (High-Level Differences)  
   - **Logistic**:  
     - Provides **probabilistic** outputs \(\sigma(z)\).  
     - Minimizes cross-entropy.  
     - Is robust to noisy data (non-separable).  
   - **SVC**:  
     - Provides a **margin-based** approach (distance from boundary).  
     - Minimizes \(\|\mathbf{w}\|^2\) subject to constraints.  
     - For perfectly separable data, it can yield zero training error with maximum margin.  
   - Both can be extended to handle **nonlinear** data with kernel methods (later expansions).

---

### 4. Appendices & Extra Details

1. **Multiclass Classification**  
   - Logistic: use **softmax** for probabilities over multiple classes. Minimizes **multiclass cross-entropy**.  
   - SVM: do “one-vs-rest” or “one-vs-one” for multiple classes.

2. **Gradient Descent Practicalities**  
   - Evaluate loss and gradient for each iteration.  
   - Possibly shuffle data for SGD.  
   - Learning rate schedules or advanced optimizers help in large-scale problems.

3. **SVC & Slack Variables** (Mentioned If Data Not Perfectly Separable)  
   - In realistic scenarios, data might overlap in feature space. Then we use “soft margin” SVC with slack \(\xi_n\) to allow misclassifications. The optimization becomes  
     \[
       \min_{\mathbf{w},\,\{\xi_n\}}\;\|\mathbf{w}\|^2 + C\sum_{n=1}^N \xi_n
       \quad\text{ subject to }v_n(\mathbf{w}^\top \mathbf{x}_n)\ge 1-\xi_n,\;\xi_n\ge 0.
     \]
   - The **constant** \(C\) balances margin size vs. misclassification penalty.

4. **Comparison**: Soft SVC vs. Logistic Regression  
   - Both yield **linear** boundaries for binary classification.  
   - SVC focuses on margin (geometric view), logistic on probabilities (statistical view).  
   - Performance often similar in practice, though details differ for outliers and data distribution assumptions.

5. **Confidence** in **Logistic vs. SVC**  
   - **Logistic**: confidence ~ predicted probability \(\sigma(z)\).  
   - **SVC**: confidence ~ distance from decision boundary (margin).  

---

### 5. Study/Exam Preparation Tips

1. **Master Gradient Descent**  
   - Understand the step formula thoroughly.  
   - Practice deriving partial derivatives (e.g. for logistic cross-entropy).  
   - Know differences between batch, mini-batch, and stochastic variants.

2. **Cross-Entropy**  
   - Recognize how it arises from maximum likelihood for Bernoulli-labeled data.  
   - Understand why it’s friendlier to gradient-based methods than 0–1 loss.

3. **Confidence & SVC**  
   - Geometric intuition: margin is the “buffer zone” around the decision boundary.  
   - Hard vs. soft margin.  
   - Understand primal formulation constraints (\(v_n\,\mathbf{w}^\top\mathbf{x}_n\ge 1-\xi_n\)), though the full dual approach is often in **Lecture 6**.

4. **Logistic vs. SVC**  
   - Both yield linear decision boundaries.  
   - Probability interpretation vs. margin interpretation.  
   - Each has pros/cons depending on data noise, interpretability, and domain constraints.