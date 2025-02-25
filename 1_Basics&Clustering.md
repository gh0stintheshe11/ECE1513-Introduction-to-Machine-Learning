# Unit 1: ML Preliminaries & Clustering

### 1. ML as a Data-Driven Approach

1. **Definition of Machine Learning**  
   - ML is a set of **data-driven approaches** that helps us understand and generalize the behavior of an environment.  
   - As highlighted in the slides: “**. . . the study of computer algorithms that improve automatically through experience**” or a type of applied statistics focusing on data.

2. **Ohm’s Law Analogy**  
   - *Historical example (1827):* Georg Ohm observed the relationship between current and voltage.  
   - He **collected data**, **hypothesized** a mathematical model (I=V/R), and **fit** that model to the data.  
   - This is effectively the same three-step process we do in ML: gather data → propose model → tune model to fit data.

3. **The Three Key Components**  
   1. **Data**: A set of collected samples.  
   2. **Model**: A function f that tries to capture how the data are generated or structured.  
   3. **Learning Algorithm**: A procedure that uses the data to find the “best” model parameters.

4. **Types of Learning Tasks**  
   - **Unsupervised Learning**: Data are unlabeled (e.g., clustering).  
   - **Supervised Learning**: Data points come with labels (e.g., classification, regression).  
   - **Reinforcement Learning**: Learning actions in an environment with rewards/penalties over time.  

---

### 2. Clustering (Unsupervised Learning)

Clustering is a core example of unsupervised learning. We want to group data points into clusters with no labels given.

1. **Basic Setup**  
   - Data set D = {x₁, x₂, …, xₙ} with x in ℝ^d.  
   - The **model** partitions data into K clusters, each associated with some representative (like a centroid).

2. **K-Means Clustering**  
   - **Goal**: Split data into K groups such that points in the same group are close to each other, measured by Euclidean distance to a centroid.  
   - **Centroids**: μₖ for cluster k=1,…,K.  
   - **Cluster Assignment**: A function f(x) = k picks which cluster a point x belongs to.

#### K-Means Algorithm Steps

```
1. Initialize K centroids μ1, ..., μK (often randomly).
2. Repeat until convergence:
   a. Cluster_Assignment:
      - For each xₙ, assign it to the centroid μₖ that is closest in Euclidean distance.
   b. Centroid_Update:
      - Update each μₖ to be the average of all points assigned to cluster k.
3. End when centroids do not move or assignments no longer change.
```

- **Always Converges**: K-Means stops at a (local) minimum of its objective function.  
- **Local Minima**: Different initializations can yield different results (run multiple times).  
- **Hyperparameter K**: Must be chosen—possibly with domain knowledge or “elbow” method.  
- **Sensitivity**: The algorithm can be sensitive to outliers or rescaling.

#### Risk (Distortion) Function  
- We define a “risk” or “distortion” J measuring how far points are from their assigned centroids:  
  \[
    J = \frac1N \sum_{n=1}^N \sum_{k=1}^K r_{n,k}\,\|\,x_n - \mu_k\|^2,
  \]  
  where rₙ,ₖ=1 if xₙ is assigned to cluster k, else 0.  
- Minimizing J is equivalent to the iterative procedure of K-Means.

#### Probabilistic View  
- Slides also connect K-Means to a **Gaussian assumption** with fixed identity covariance. Minimizing sum of squared distances is like maximizing the likelihood if each cluster is an N(μₖ, I).

---

### 3. Soft K-Means & Extensions

1. **Soft K-Means**  
   - Instead of hard 0/1 assignments (rₙ,ₖ ∈ {0,1}), each xₙ has a “degree of belonging” to each cluster k.  
   - Formula uses an exponential weighting:  
     \[
       r_{n,k} = \frac{\exp(-\beta\|x_n - \mu_k\|^2)}{\sum_{j=1}^K \exp(-\beta\|x_n - \mu_j\|^2)},
     \]  
   - As β→∞, it approximates regular K-Means.

2. **Segmentation & More**  
   - Slides show an **image segmentation** example: each pixel is a point in ℝ³ (RGB).  
   - K-Means groups them so that pixels in the same cluster share a centroid color.

---

### 4. Notion of Risk & Convergence

1. **Objective**  
   - The K-Means algorithm is effectively doing risk minimization, with a straightforward geometric interpretation.

2. **Convergence**  
   - Each iteration reduces the total distortion J, so it cannot “worsen.”  
   - Once assignments and centroids stabilize, it stops.

3. **Local Minima**  
   - Because it’s not a convex problem in all parameters simultaneously, K-Means can get stuck in local minima.  
   - Common strategy: Run multiple initializations, pick the best final solution.

---

### 5. Putting It All Together

- **Unsupervised** = No labels. K-Means is a prime example.  
- The **“Data→Model→Algorithm”** approach from the slides:  
  - **Data**: D = {x₁,…,xₙ}, no labels.  
  - **Model**: f that assigns xₙ to a cluster ID in {1,…,K}.  
  - **Learning Algorithm**: K-Means (or Soft K-Means, or other methods).  

- **Transition**: Future lectures (like Lecture 2) discuss distribution fitting, and Lecture 4+ move to **supervised** tasks.

---

### 6. Quick Reference & Study Tips

1. **ML is iterative**: We often begin with data, guess a model, refine with an algorithm, check results, etc.  
2. **K-Means**:  
   - Key steps: assign points, update centroids, repeat.  
   - Minimizes average squared distance to centroids.  
   - Always converges but not necessarily globally optimal.  
3. **Soft K-Means**: A “fuzzy” extension with fractional assignments.  
4. **Hyperparameter K**: Must be set (or tested) by the user.  
5. **Practical Pitfalls**: outliers, feature scaling, initialization.  

**Extra Mentions from Slides**  
- Real-world examples:  
  - **Bank record** example (monthly transactions vs number of transactions).  
  - **Pets** (supervised vs unsupervised data explanation).  
  - The slides mention we cluster data naturally (plants vs animals, etc.).  
- The final slides emphasize that **K-Means** is just one approach. We can continue to other tasks like distribution estimation, dimensionality reduction, or supervised learning.