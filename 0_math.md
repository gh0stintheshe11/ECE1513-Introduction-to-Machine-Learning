# 1. Vectors and Their Operations

## 1.1 Notation for Vectors

- We often write a vector $\mathbf{x}$ as a column:

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
$$

  - Example: If $\mathbf{x}$ is 3-dimensional, it might be

$$
\mathbf{x} = \begin{bmatrix} 2 \\ 5 \\ -1 \end{bmatrix}
$$

- A **row vector** is just the transpose: $\mathbf{x}^T = [x_1 \; x_2 \; \dots \; x_n]$.

## 1.2 Adding Vectors

- If $\mathbf{u}$ and $\mathbf{v}$ are both $n$-dimensional, you add them **component by component**:

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}
$$
  
- **Example**:

$$
\mathbf{u} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}, \quad \mathbf{v} = \begin{bmatrix} -1 \\ 5 \end{bmatrix}
$$

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} 2 + (-1) \\ 3 + 5 \end{bmatrix} = \begin{bmatrix} 1 \\ 8 \end{bmatrix}
$$

## 1.3 Scaling a Vector (Multiplying by a Number)

- If $c$ is a scalar (regular real number), then

$$
c \mathbf{v} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \\ \vdots \\ c \cdot v_n \end{bmatrix}
$$
  
- **Example**:

$$
\mathbf{v} = \begin{bmatrix} 4 \\ -2 \end{bmatrix} \quad\text{and}\quad c=3
$$

$$
c\mathbf{v} = 3\mathbf{v}= \begin{bmatrix} 3 \times 4 \\ 3 \times (-2) \end{bmatrix} = \begin{bmatrix} 12 \\ -6 \end{bmatrix}
$$

## 1.4 Dot (Inner) Product

- The **dot product** of two $n$-dimensional vectors $\mathbf{u}, \mathbf{v}$ is:

$$
\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = u_1 v_1 + u_2 v_2 + \dots + u_n v_n
$$
  
- **Example**: If $\mathbf{u} = [1, 3]^T$ and $\mathbf{v} = [2, -4]^T$,

$$
\mathbf{u} \cdot \mathbf{v} = (1)(2) + (3)(-4) = 2 + (-12) = -10
$$

## 1.5 Length (Norm) of a Vector

- The **Euclidean (2-) norm** of $\mathbf{x}$ is:

$$
\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}
$$
  
- **Example**: 

$$
\mathbf{x}= \begin{bmatrix} 2 \\ -5 \end{bmatrix}
$$

$$
\|\mathbf{x}\|_2 = \sqrt{2^2 + (-5)^2} = \sqrt{4 + 25} = \sqrt{29}
$$

## 1.6 Outer (Cross) Product

- For vectors $\mathbf{u}\in \mathbb{R}^m$ and $\mathbf{v}\in \mathbb{R}^n$, the **outer product** $\mathbf{u}\mathbf{v}^T$ is an $m\times n$ matrix with entries

$$
(\mathbf{u}\mathbf{v}^T)_{ij} = u_i v_j
$$
  
- **Example**: If 

$$
\mathbf{u} = \begin{bmatrix} 1 \\ 4 \end{bmatrix} \quad\text{and}\quad \mathbf{v} = \begin{bmatrix} 2 \\ 3 \\ -1 \end{bmatrix}
$$
  
  then $\mathbf{u}\mathbf{v}^T$ is a $2\times 3$ matrix:
  
$$
\mathbf{u}\mathbf{v}^T = \begin{bmatrix} 1\times 2 & 1\times 3 & 1\times (-1) \\ 4\times 2 & 4\times 3 & 4\times (-1) \end{bmatrix} = \begin{bmatrix} 2 & 3 & -1 \\ 8 & 12 & -4 \end{bmatrix}
$$

---

# 2. Matrices: Multiplying, Adding, and Transposing

## 2.1 Matrix Dimensions

- A matrix $A$ is often denoted as $m\times n$ if it has $m$ rows and $n$ columns.
  - **Example**: $A$ = $\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}$ is $2\times2$.

## 2.2 Matrix Addition

- Add two matrices of the same size by **adding each corresponding entry**.
  - **Example**:
  
$$
\begin{bmatrix} 1 & 3 \\ 2 & -1 \end{bmatrix} + \begin{bmatrix} 4 & 0 \\ -3 & 2 \end{bmatrix} = \begin{bmatrix} 1+4 & 3+0 \\ 2+(-3) & -1+2 \end{bmatrix} = \begin{bmatrix} 5 & 3 \\ -1 & 1 \end{bmatrix}
$$

## 2.3 Matrix-Vector Multiplication

- If $A$ is $m\times n$ and $\mathbf{x}$ is $n\times 1$, then $A\mathbf{x}$ is $m\times 1$.
  - **Component-wise**:
  
$$
(A\mathbf{x})_i = a_{i1}x_1 + a_{i2}x_2 + \dots + a_{in}x_n
$$
    
- **Example**:

$$
A = \begin{bmatrix} 2 & -1 & 0 \\ 1 & 4 & 2 \end{bmatrix} \quad \text{and} \quad \mathbf{x}= \begin{bmatrix} 3 \\ -1 \\ 5 \end{bmatrix}
$$
  
  - Then 
  
$$
A\mathbf{x} = \begin{bmatrix} 2\cdot 3 + (-1)\cdot (-1) + 0\cdot 5 \\ 1\cdot 3 + 4\cdot(-1) + 2\cdot 5 \end{bmatrix} = \begin{bmatrix} 6 + 1 + 0 \\ 3 + (-4) + 10 \end{bmatrix} = \begin{bmatrix} 7 \\ 9 \end{bmatrix}
$$

## 2.4 Matrix-Matrix Multiplication

- If $A$ is $m\times n$ and $B$ is $n\times p$, then the product $C = A B$ is an $m\times p$ matrix. The $(i,j)$-th entry is:

$$
c_{ij} = \sum_{k=1}^n a_{ik} b_{kj} = a_{i1}b_{1j} + a_{i2}b_{2j} + \dots + a_{in}b_{nj}
$$
  
- **Example**:

$$
A= \begin{bmatrix} 1 & 2 \\ 0 & -1 \end{bmatrix}, \quad B= \begin{bmatrix} 3 & 1 & 2\\ 1 & 0 & 4 \end{bmatrix}
$$
  
  - Then $C=AB$ is $2\times 3$:
  
$$
C = \begin{bmatrix} 1\cdot 3 + 2\cdot 1 & 1\cdot 1 + 2\cdot 0 & 1\cdot 2 + 2\cdot 4 \\ 0\cdot 3 +(-1)\cdot 1 & 0\cdot 1 +(-1)\cdot 0 & 0\cdot 2 +(-1)\cdot 4 \end{bmatrix}
$$

$$
= \begin{bmatrix} 3 + 2 & 1 + 0 & 2 + 8 \\ 0 + (-1) & 0 + 0 & 0 + (-4) \end{bmatrix} = \begin{bmatrix} 5 & 1 & 10 \\ -1 & 0 & -4 \end{bmatrix}
$$

## 2.5 Transpose

- The **transpose** of $A$ flips it across the diagonal, so $(A^T)_{ij} = A_{ji}$.
- **Example**:

$$
A = \begin{bmatrix} 1 & 4 & -2 \\ 3 & 0 & 1 \end{bmatrix}, \quad A^T = \begin{bmatrix} 1 & 3 \\ 4 & 0 \\ -2 & 1 \end{bmatrix}
$$

## 2.6 Identity and Inverse

- **Identity matrix** $I$ is square, with 1s on diagonal, 0s elsewhere. 
  - For any compatible $\mathbf{x}$, $I \mathbf{x} = \mathbf{x}$.
- A square matrix $A$ is **invertible** if there is a matrix $A^{-1}$ such that $AA^{-1} = I$. 

---

# 3. Eigenvalues and Eigenvectors (Step-by-Step)

## 3.1 The Basic Idea

- An **eigenvector** $\mathbf{v}$ of a square matrix $A$ is a nonzero vector so that:

$$
A\mathbf{v} = \lambda\mathbf{v}
$$

  where $\lambda$ is the **eigenvalue**.

## 3.2 How To Find Them (In 2D or 3D, for example)

1. Write $\mathbf{v} = [v_1,\dots,v_n]^T$.
2. **Eigenvalue equation**: $A\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}$.
3. This can be rearranged to $(A - \lambda I)\mathbf{v}=\mathbf{0}$.
4. For a **non-trivial** (nonzero) $\mathbf{v}$ to exist, we need 

$$
\det(A - \lambda I)=0
$$

5. Solve that **determinant** equation to find possible $\lambda$. Then find $\mathbf{v}$ for each $\lambda$.

## 3.3 Quick 2D Example

- Let

$$
A = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}
$$
  
1. The characteristic equation for $\lambda$:

$$
\det \begin{bmatrix} 2-\lambda & 1 \\ 0 & 3-\lambda \end{bmatrix} = (2-\lambda)(3-\lambda) = 0
$$
   
2. So $\lambda$ solutions are $\lambda_1=2$ and $\lambda_2=3$.

3. **Eigenvector for $\lambda_1=2$**: solve $A\mathbf{v}=2\mathbf{v}$.

$$
\begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 2v_1 \\ 2v_2 \end{bmatrix}
$$
   
   - First row says: $2 v_1 + 1\cdot v_2 = 2v_1 \implies v_2= 0$.
   - So $\mathbf{v} = \begin{bmatrix}1\\0\end{bmatrix}$ (scaled in any nonzero multiple).
   
4. **Eigenvector for $\lambda_2=3$**:

$$
A\mathbf{v} = 3\mathbf{v}
$$
   
   - First row: $2v_1 + v_2 = 3v_1 \implies v_2 = v_1$.
   - So $\mathbf{v}= \begin{bmatrix}1\\1\end{bmatrix}$ is an eigenvector (again, scaling is allowed).

---

# 4. Simple Derivatives With Vectors

## 4.1 Sum of Components

- If you see $\sum_{i=1}^n x_i$, it's the same as "x1 + x2 + … + xn". For example,  

$$
x_1 + x_2 + \dots + x_n
$$

## 4.2 Gradient Examples

- For a function $f(\mathbf{w}) = \|\mathbf{w}\|^2$ = $(w_1^2 + w_2^2 + \dots + w_d^2)$,

$$
\nabla_{\mathbf{w}} f(\mathbf{w}) = \begin{bmatrix} 2w_1 \\ 2w_2 \\ \vdots \\ 2w_d \end{bmatrix}
$$
  
- For $(1/2)\|\mathbf{w}\|^2 = (1/2)(w_1^2 + \dots + w_d^2)$, the gradient would be $[w_1, w_2, \dots, w_d]^T$, etc.

## 4.3 Linear Regression Example

- Suppose the cost is 

$$
R(\mathbf{w}) = \sum_{n=1}^N [(\mathbf{w}^T \mathbf{x}_n) - v_n]^2
$$
  
  If we define $\mathbf{X}$ to group the data, this is $\|\mathbf{X}^T \mathbf{w} - \mathbf{v}\|^2$.
  
- The gradient with respect to $\mathbf{w}$ typically ends up $\sim \mathbf{X}(\mathbf{X}^T\mathbf{w} - \mathbf{v})$.

---

# 5. Examples and "Panic-proof" Formulas

1. **Sum of a list**: $x_1 + x_2 + \dots + x_n$  
2. **Dot product** of vectors $\mathbf{u}$ and $\mathbf{v}$:

$$
u_1v_1 + u_2v_2 + \dots + u_nv_n
$$
   
3. **2D determinant**: if 

$$
A= \begin{bmatrix} a & b\\ c & d \end{bmatrix}
$$
   
   then $\det(A)=ad - bc$.
   
4. **Distance** between $\mathbf{x}$ and $\mathbf{y}$:

$$
\|\mathbf{x}-\mathbf{y}\| = \sqrt{(x_1 - y_1)^2 + (x_2-y_2)^2 + \dots + (x_n-y_n)^2}
$$

---

# 6. Putting It All Together

- **In K-Means**: you repeatedly compute distances $\|\mathbf{x}_n - \mathbf{\mu}_k\|^2$ for each point $\mathbf{x}_n$ and centroid $\mathbf{\mu}_k$. Then you average points to update centroids:  

$$
\mathbf{\mu}_k = \frac{1}{\text{(number of points in cluster k)}} \sum_{ \mathbf{x}_n \in \text{cluster k}} \mathbf{x}_n
$$
  
- **In PCA**: you find the sample covariance matrix 

$$
\Sigma = \frac{1}{N}\sum_{n=1}^N (\mathbf{x}_n - \bar{\mathbf{x}})(\mathbf{x}_n-\bar{\mathbf{x}})^T
$$
  
  and get eigenvectors of $\Sigma$.
  
- **In Linear Regression**: you might do a normal equation or do gradient steps.

---

1. Keep these **"no-summation"** forms in mind:
   - Dot product:  $(u_1)(v_1) + (u_2)(v_2) + \dots + (u_n)(v_n)$.
   - Distances:  $\sqrt{ (x_1 - y_1)^2 + \dots + (x_n - y_n)^2 }$.
   
2. For **eigen** problems in 2D, you can quickly do the "$(2-\lambda)(3-\lambda) - \text{(whatever)} = 0$" approach. 

3. Remember, **practice** small numeric examples. If you get stuck, write everything out as $x_1 + x_2 + \dots + x_N$.

---

# 7. Probability Theory

## 1. Probability of Discrete Outcomes

1. **Discrete Random Variable**  
   - Example: a Bernoulli random variable $X$ can take the values 0 or 1.  
   - We define "$P(X=1) = \theta$" and "$P(X=0) = 1 - \theta$." 
   - If you see $\sum_{x}$, just read it as "x1 + x2 + …" for all possible x.  

2. **Probability of Multiple Independent Observations**  
   - Suppose we observe $N$ independent results $x1, x2, \dots, xN$.  
   - If each $x_i$ is Bernoulli($\theta$), then

$$
P(\text{all data}) = \theta^k (1-\theta)^{N-k}
$$

Where $k$ is the number of 1's and $N-k$ is the number of 0's.

   - Example: data = {1,1,0,1}, then the probability is $\theta^3 \times (1-\theta)^1$.

## 2. Probability for Continuous Outcomes

1. **Continuous Random Variable**  
   - Example: a Gaussian/Normal random variable $X$ can take any real value.  
   - Probability is given by a **PDF** (probability density function) $p(x)$.  
   - If you see $\int p(x)\,dx$, think of it as "the area under the curve of p(x)" is 1.  

2. **Gaussian (Normal) Distribution**  
   - For 1D, $X\sim \text{N}(\mu, \sigma^2)$, the PDF is

$$
p(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-(x-\mu)^2 / (2\sigma^2)\right)
$$

   - Key expansions:
     1. "$\sqrt{(2)(\pi)(\sigma^2)}$" is "$\sqrt{2}\times\sqrt{\pi}\times\sigma$."
     2. The exponent $(x-\mu)^2/(2\sigma^2)$ is just "( (x minus mu) times (x minus mu) ) / (2 times sigma^2)."

## 3. Likelihoods and Log-Likelihoods (Maximum Likelihood Estimation)

1. **Likelihood** = Probability of Data, Treated as a Function of Unknown Parameters  
   - If data = {$x_1, x_2, \dots, x_N$}, each observation is from the same distribution **independently**.  
   - Then the probability is 

$$
\prod_{n=1}^N P_\theta(x_n)
$$

   which is "$P_\theta(x_1)\times P_\theta(x_2)\times \dots \times P_\theta(x_N)$."

2. **Log Trick**  
   - Often we take the **log** of that product (the "log-likelihood"), because "log of (a product) is the sum of logs."  
   - So instead of $\prod$, we do $x_1 + x_2 + \dots$ for the logs.  
   - Example: if $x_n$ are Bernoulli($\theta$), 

$$
\ln L(\theta) = \sum_{n=1}^N [x_n \ln(\theta) + (1 - x_n)\ln(1-\theta)]
$$

   (But in an exam panic you can rewrite it as "$(x_1)(\ln \theta)+(1 - x_1)(\ln(1-\theta)) + (x_2)(\ln \theta)+(1 - x_2)(\ln(1-\theta)) + \dots$")

3. **Finding MLE**: "Set derivative = 0" approach  
   - Because if the derivative of log-likelihood w.r.t. $\theta$ is zero, that's usually your candidate for max.  
   - **Example**: If data = {1,2,3} from an **exponential** distribution with parameter $\lambda$, the likelihood is $\lambda^N \exp(-\lambda(x_1 + x_2 + x_3 + \dots + x_N))$.  
   - Then you do log-likelihood = "$N \ln \lambda - \lambda$ times $(x_1 + x_2 + x_3 + \dots)$."  
   - Set derivative w.r.t. $\lambda$ = 0, solve for $\lambda$.

## 4. Common Distributions (No Summation Notation)

1. **Bernoulli($\theta$)**:  
   - "$P(X=1) = \theta,\;P(X=0)=1-\theta$."

2. **Exponential($\lambda$)**:  
   - "$p(x)= \lambda \exp(-\lambda x)$" for $x \ge0$.  

3. **Gaussian($\mu,\sigma^2$)**:  
   - "$p(x)= [1/( \sqrt{2\pi}\sigma)] \exp( - (x-\mu)^2/(2\sigma^2) )$."

## 5. A Quick Example to Put It All Together

**Example**: Data = {1,2,3} from an Exponential distribution with unknown rate $\lambda$.

1. **Write Likelihood**:

$$
L(\lambda) = \lambda^3\exp(-\lambda\times(1 + 2 + 3))
$$

   because the sum of $x_1 + x_2 + x_3$ is 6 here.

2. **Log-likelihood**:

$$
\ln L(\lambda) = 3\ln(\lambda) - \lambda (1 + 2 + 3)
$$

   Expand that out: "(3 times log lambda) minus lambda times 6."

3. **Set Derivative to 0**:
   - derivative = "$(3 / \lambda) - 6 = 0$." 
   - Solve: "$(3 / \lambda) = 6 \Rightarrow \lambda = (3/6) = 0.5$."

So MLE is $\lambda^* = 0.5$.

---

# Wrapping Up

- The "nasty" parts of probability math often revolve around **likelihoods** (especially with exponentials, logs, products). But you can always treat "$\prod_{n=1}^N$" as "(stuff1) times (stuff2) times … (stuffN)," and "$\sum_{n=1}^N$" as "stuff1 + stuff2 + … + stuffN." 
- The main distributions used in the lectures (Bernoulli, Exponential, Gaussian) all revolve around these expansions. 
- **Stay calm**: Expand step by step. For instance, if you see $\prod_{n=1}^N P_\theta(x_n)$, literally rewrite it as "$P_\theta(x_1) \times P_\theta(x_2) \times \dots P_\theta(x_N)$" so you see each factor. Then take a **log** to turn it into additions.  
- For MLE: you typically do "**(1) Write log-likelihood** => **(2) expand** => **(3) derivative w.r.t. parameter** => **(4) set = 0** => solve."