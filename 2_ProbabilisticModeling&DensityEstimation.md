## Unit 2: Probabilistic Modeling & Density Estimation

### 1. Why Probabilistic Modeling?

1. **Motivation: Outlier Detection & Beyond**  
   - We often need to assess whether new observations are “normal” or “unusual.”  
   - A principled way: **Estimate a probability distribution** for normal data, then check how likely new samples are under that distribution.  
   - Goes beyond simple clustering—**density estimation** helps with anomaly detection, generative modeling, and more.

2. **Unsupervised Setting**  
   - Data are unlabeled: D = {x₁, x₂, …, xₙ}.  
   - We hypothesize a **distribution** Pθ(x) with unknown parameter(s) θ.  
   - Goal: Learn the **best** θ so that Pθ(x) fits the data well (i.e., assign high probability to points like xₙ in D).

---

### 2. Probability Theory Basics

1. **Discrete Random Variables**  
   - x takes values in a finite or countable set (e.g., {0,1} or {0,1,2,…}).  
   - Probability mass function (PMF): P(x=aᵢ).  
   - Must satisfy:  
     \[
       \sum_i P(a_i) = 1,\quad 0 \leq P(a_i) \leq 1.
     \]
   - **Bernoulli(θ)**: x ∈ {0,1} with P(x=1)=θ and P(x=0)=1−θ.

2. **Continuous Random Variables**  
   - x ∈ ℝ or ℝ^d. Probability density function (PDF) P(x)≥0, and \(\int P(x)\,dx = 1.\)  
   - **Gaussian**(μ,σ²):  
     \[
       P(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Bigl(-\frac{(x-\mu)^2}{2\sigma^2}\Bigr).
     \]
   - **Exponential**(λ):  
     \[
       P(x)=\lambda e^{-\lambda x},\; x\ge0.
     \]

3. **Independence & i.i.d.**  
   - x₁,…,xₙ are **independent & identically distributed** if they come from the same distribution P(x) with no dependence on each other.

4. **Expectation** (Mean)  
   - Discrete: E[f(x)] = ∑ₓ f(x) P(x).  
   - Continuous: E[f(x)] = ∫ f(x) P(x) dx.  
   - For i.i.d. samples, sample mean approximates E[x].

---

### 3. Maximum Likelihood Estimation (MLE)

**Core concept**: If we assume the data are i.i.d. from Pθ(x), how do we find θ?

1. **Likelihood** L(θ)  
   - For dataset D = {x₁, …, xₙ}:  
     \[
       L(\theta)=P_\theta(D)=\prod_{n=1}^N P_\theta(x_n).
     \]
   - Often easier to use **log-likelihood**:  
     \[
       \ln L(\theta)=\sum_{n=1}^N \ln P_\theta(x_n).
     \]

2. **Finding θ* by MLE**  
   - \(\theta^* = \arg\max_\theta L(\theta)\) or equivalently \(\arg\max_\theta \ln L(\theta).\)  
   - Take derivative w.r.t. θ, set to zero → solve.  
   - **Examples**:  
     - **Bernoulli(θ)**:  
       \[
         \ln L(\theta)=\sum_{n=1}^N [x_n \ln \theta + (1-x_n)\ln(1-\theta)],
       \]
       solution: \(\theta^*=\frac{\text{(Number of 1’s)}}{N}.\)  
     - **Gaussian(μ,σ²)** (1D): \(\mu^*\) = sample mean, \(\sigma^{2*}\) = sample variance.  
     - **Exponential(λ)**: \(\lambda^*=N/\sum_n x_n.\)

3. **Interpretation**  
   - MLE picks the θ that makes the observed data **most probable** under Pθ.

---

### 4. Connecting Clustering & Maximum Likelihood

1. **K-Means = Simple Gaussian Fitting**  
   - If we assume each cluster is a **Gaussian** with mean μₖ and *fixed identity covariance*, then MLE for μₖ gives:  
     \[
       \mu_k = \text{mean of points in cluster }k.
     \]
   - **Assign each xₙ to whichever μₖ** has largest P( xₙ | that cluster ), which is the same as **smallest Euclidean distance**.  
   - The iterative method: **K-Means** is effectively an **EM**-like approach for these means.

2. **Soft K-Means** & **GMM**  
   - Instead of hard assignment, each point belongs fractionally to clusters.  
   - More general is **Gaussian Mixture Model (GMM)**:  
     \[
       P(x)=\sum_{k=1}^K \pi_k\, \mathcal{N}(x|\mu_k,\Sigma_k),
     \]  
     with mixing weights πₖ, means μₖ, and covariance matrices Σₖ.  
   - MLE for GMM is done via the **Expectation-Maximization (EM)** algorithm.  
   - **Soft K-Means** is just a special case with Σ fixed = identity and πₖ=1/K.

---

### 5. Model Checking & Validation

1. **Overfitting**  
   - If the model is too flexible (e.g. too many parameters or mixture components), it might “memorize” data but fail to generalize.

2. **Data Splitting**  
   - **Train**: Fit parameters (e.g. θ or the centroids).  
   - **Validation**: Check performance, tune hyperparameters (like number of clusters K).  
   - **Test**: Final check to see how well the model generalizes.

3. **Likelihood on Test Set**  
   - For density estimation, we can measure log-likelihood on the test set T. If it’s much lower than on training data, we might be overfitting.

4. **Outlier Detection**  
   - If Pθ(x_new) is very low, we can label x_new as an “outlier.”  
   - Similarly, large negative log-likelihood indicates unusual data point.

---

### 6. Step-by-Step Examples

1. **Exponential Distribution**  
   - Example from Lecture: D={1,2,3}.  
   - L(λ)=λ^N exp(−λ∑ₙ xₙ).  
   - lnL(λ)=N lnλ − λ∑ₙ xₙ.  
   - Solve derivative=0 → λ^*=N/∑ₙ xₙ = 3/(1+2+3)=0.5.

2. **Connection to K-Means**  
   - Suppose each cluster has identity covariance Σ=I. The negative log-likelihood of xₙ under cluster k is proportional to ‖xₙ−μₖ‖². Minimizing total negative log-likelihood = Minimizing sum of squared distances → The K-Means objective function.

3. **Gaussian**(μ,σ²) MLE**  
   - For 1D data:  
     - μ^* = average of xₙ.  
     - σ^{2*} = average of (xₙ−μ^*)².

---

### 7. Key Takeaways / “No Detail Too Small”

1. **Data→Distribution**: We treat each sample xₙ as a random draw from Pθ(x).  
2. **i.i.d.**: Usually crucial for ML to handle them as independent draws.  
3. **Max Likelihood**: Standard tool for parameter estimation (θ^* solves arg max Pθ(D)).  
4. **Log-likelihood** is used for convenience (sums vs. products).  
5. **MLE for Common Distributions**:  
   - Bernoulli(θ): fraction of 1’s.  
   - Gaussian(μ,σ²): sample mean & variance.  
   - Exponential(λ): inverse of sample mean.  
6. **Link to Clustering**:  
   - K-Means can be seen as MLE for means of K Gaussians with fixed identity covariance.  
   - Soft K-Means → partial membership → a simpler version of GMM.  
7. **Testing & Validation**:  
   - Data splitting helps avoid overfitting.  
   - Evaluate log-likelihood or related metrics (like distortion in K-Means) on validation/test sets.

---

### 8. Frequently Confused Points

1. **MLE vs. MAP**: The Lecture focuses on **MLE** (maximum likelihood). A more advanced approach is **MAP** (maximum a posteriori) which would incorporate priors. Not covered deeply here, but keep in mind for advanced classes.  
2. **Covariance in K-Means**: K-Means effectively sets Σ=I; real data might need full Σ → that’s GMM.  
3. **Outlier vs. Low Probability**: “Unusual” means Pθ(x_new) is **small**. Perfect for outlier detection but you must ensure your distribution is well-fitted.

---

## Quick-Reference Formulae

1. **Log-Likelihood** for i.i.d. data:  
   \[
     \ln L(\theta) = \sum_{n=1}^N \ln P_\theta(x_n).
   \]

2. **Bernoulli(θ)**:  
   \[
     \theta^* = \frac{\#\text{(ones)}}{N}.
   \]

3. **Exponential(λ)**:  
   \[
     \lambda^* = \frac{N}{\sum_n x_n}.
   \]

4. **Gaussian(μ, σ²)** (1D):  
   \[
     \mu^* = \frac1N \sum_{n=1}^N x_n,\quad
     \sigma^{2*} = \frac1N \sum_{n=1}^N (x_n-\mu^*)^2.
   \]

5. **K-Means Distortion**:  
   \[
     J = \frac1N \sum_{n=1}^N \sum_{k=1}^K r_{n,k}\|x_n - \mu_k\|^2,
   \]
   with rₙ,ₖ=1 if point xₙ assigned to cluster k, else 0.