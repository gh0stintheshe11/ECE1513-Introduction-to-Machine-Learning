# **Unit 2: Probabilistic Modeling and Density Estimation**

---

## **1. General Idea**
### **What problem are we solving?**
- **Goal:** Estimate the **parameters of a probability distribution** from given data.  
- **Key Idea:** We assume that our data is generated from an **unknown probability distribution** and want to **find the best parameter values** that fit this distribution.
- **Approach:**  
  - We use **Maximum Likelihood Estimation (MLE)** to find the parameters that maximize the probability of observing our dataset.  
  - MLE is commonly used for **density estimation, classification models, and regression**.

### **What models are used?**
1. **Exponential Distribution**:  
   - Used to model waiting times between events.
   - Probability Density Function (PDF):  
     \[
     P(x; \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0
     \]
   - We estimate **\( \lambda \)** using MLE.
  
2. **Bernoulli Distribution** (Binary Classification):  
   - Used for data with **two possible outcomes** (0 or 1).
   - Probability Mass Function (PMF):  
     \[
     P(v_n | x_n, w) = \sigma(w^T x_n)^{v_n} (1 - \sigma(w^T x_n))^{(1 - v_n)}
     \]
   - We estimate **\( w \)** (weights of the classifier) using MLE.

3. **Gaussian Distribution**:  
   - Used to model **continuous** data.
   - Probability Density Function:  
     \[
     P(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
     \]
   - We estimate **\( \mu \) (mean) and \( \sigma^2 \) (variance)** using MLE.

---

## **2. Definitions**
These are the key terms and definitions you must know.

| **Term**                | **Definition** |
|------------------------|--------------|
| **Likelihood Function \( L(\theta) \)** | The probability of the observed data given parameter \( \theta \):  \( L(\theta) = P(D | \theta) \). |
| **Log-Likelihood Function \( \ell(\theta) \)** | The natural logarithm of the likelihood function: \( \ell(\theta) = \ln L(\theta) \). |
| **Maximum Likelihood Estimation (MLE)** | The method to estimate parameters by maximizing the likelihood function. |
| **Probability Density Function (PDF)** | A function that gives the probability of a continuous random variable falling within a range. |
| **Probability Mass Function (PMF)** | A function that gives the probability of a discrete random variable taking a specific value. |
| **Log Trick** | Since likelihoods involve products, taking the logarithm converts them into sums, making differentiation easier. |

---

## **3. Solution Process**
This section provides **clear, step-by-step** derivations for **MLE for different distributions**.

### **Step 1: Maximum Likelihood for Exponential Distribution**
#### **Problem:** Given data **\( D = \{ x_1, x_2, ..., x_N \} \)** drawn from an **Exponential Distribution**, estimate \( \lambda \).

1. **Write the likelihood function**  
   \[
   L(\lambda) = \prod_{n=1}^{N} P(x_n | \lambda)
   \]
   Substituting the exponential PDF:
   \[
   L(\lambda) = \prod_{n=1}^{N} \lambda e^{-\lambda x_n}
   \]
   \[
   = \lambda^N e^{-\lambda \sum x_n}
   \]

2. **Take the log-likelihood**  
   Using the **log trick**:
   \[
   \ell(\lambda) = \ln L(\lambda) = N \ln \lambda - \lambda \sum x_n
   \]

3. **Differentiate and set to zero**  
   \[
   \frac{d}{d\lambda} \ell(\lambda) = \frac{N}{\lambda} - \sum x_n = 0
   \]

4. **Solve for \( \lambda \)**
   \[
   \lambda^* = \frac{N}{\sum x_n}
   \]

---

### **Step 2: Maximum Likelihood for Bernoulli Distribution (Binary Classification)**
#### **Problem:** Given binary labels **\( v_n \in \{0,1\} \)**, estimate \( \theta \) (probability of success).

1. **Write the likelihood function**  
   \[
   L(\theta) = \prod_{n=1}^{N} \theta^{v_n} (1 - \theta)^{(1 - v_n)}
   \]

2. **Take the log-likelihood**  
   \[
   \ell(\theta) = \sum_{n=1}^{N} v_n \ln \theta + (1 - v_n) \ln (1 - \theta)
   \]

3. **Differentiate and set to zero**  
   \[
   \frac{d}{d\theta} \ell(\theta) = \frac{\sum v_n}{\theta} - \frac{N - \sum v_n}{1 - \theta} = 0
   \]

4. **Solve for \( \theta \)**
   \[
   \theta^* = \frac{\sum v_n}{N}
   \]

---

### **Step 3: Maximum Likelihood for Gaussian Distribution**
#### **Problem:** Given data \( D = \{x_1, x_2, ..., x_N\} \), estimate \( \mu \) and \( \sigma^2 \).

1. **Write the likelihood function**  
   \[
   L(\mu, \sigma^2) = \prod_{n=1}^{N} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_n - \mu)^2}{2\sigma^2}}
   \]

2. **Take the log-likelihood**  
   \[
   \ell(\mu, \sigma^2) = -\frac{N}{2} \ln (2\pi \sigma^2) - \frac{1}{2\sigma^2} \sum (x_n - \mu)^2
   \]

3. **Differentiate w.r.t \( \mu \) and \( \sigma^2 \), then solve for parameters**  
   \[
   \mu^* = \frac{1}{N} \sum x_n, \quad \sigma^{2*} = \frac{1}{N} \sum (x_n - \mu^*)^2
   \]

---

## **4. Sample Numerical Example**
**Given dataset:** \( D = \{1, 2, 3\} \), assuming an **Exponential Distribution**.

1. **Write the likelihood function:**  
   \[
   L(\lambda) = \lambda^3 e^{- \lambda (1+2+3)}
   \]

2. **Take the log-likelihood:**  
   \[
   \ell(\lambda) = 3 \ln \lambda - 6\lambda
   \]

3. **Differentiate and solve:**  
   \[
   \frac{3}{\lambda} - 6 = 0 \Rightarrow \lambda^* = \frac{3}{6} = 0.5
   \]

✔ **Correct MLE estimate: \( \lambda^* = 0.5 \).**

---

## **5. Other Important Details**
- **MLE Assumptions**  
  - The data samples are **independent and identically distributed (i.i.d.)**.
  - The model follows a known **probability distribution**.

- **Log Trick**  
  - Using logarithms simplifies likelihood calculations by **converting products into sums**.

- **Connection Between MLE and Cross-Entropy**  
  - In **classification**, MLE is equivalent to **minimizing Cross-Entropy Loss**.