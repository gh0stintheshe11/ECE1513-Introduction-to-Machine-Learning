## Unit 1: Machine Learning Basics & Clustering

### 1. Key Machine Learning Concepts

**Machine Learning Definition**: 
- Data-driven approaches that help us understand the environment and generalize it
- Learns patterns from data rather than being explicitly programmed

**Three Components of ML**:
1. **Data**: 
   - For unsupervised learning: D = {x1, x2, ..., xN}
   - For supervised learning: D = {(x1,v1), (x2,v2), ..., (xN,vN)}
   - Data can be in any dimension (d)

2. **Model**: 
   - Function that captures patterns (f: X → Y)
   - Examples: f(x) = wx (linear), f(x) = cluster assignment, etc.
   - Lives in a hypothesis space H (set of all possible models)

3. **Learning Algorithm**: 
   - Finds optimal model parameters from data
   - Often involves minimizing some error/risk function
   - Outputs f* (the "best" model from H based on data)

**Types of Learning Tasks**:
- **Unsupervised Learning**: No labels (clustering, dimensionality reduction, density estimation)
- **Supervised Learning**: Labeled data (regression, classification)
- **Reinforcement Learning**: Learning from rewards/penalties over time

**Risk Minimization Framework**: 
- **Loss Function**: L(y,v) measures error between prediction y and actual v
- **Risk**: Expected Loss = E{L(y,v)} (theoretical, based on true distribution)
- **Empirical Risk**: (1/N) * [L(y1,v1) + L(y2,v2) + ... + L(yN,vN)] (measured on dataset)
- **Optimal Model**: f* = argmin(f∈H) R(f)

### 2. Clustering

**Goal**: Group similar data points together, discover underlying structures

**K-Means Clustering**:
- Partitions data into K clusters, each represented by a centroid
- Model assigns each point x to nearest centroid
- Objective: Minimize total squared distance between points and their centroids

**K-Means Algorithm**:
```
1. Initialize K centroids μ1,...,μK (randomly or using heuristics)
2. Repeat until convergence:
   a. Cluster_Assignment: Assign each point to nearest centroid
      rn,k = 1 if k = argmin‖xn-μj‖, else 0
   b. Centroid_Update: μk = (sum of all points in cluster k)/(number of points in cluster k)
3. Return μ1,...,μK
```

**Distortion/Risk Function**: 
- J = (1/N) * sum(sum(rn,k * ‖xn-μk‖²))
- Represents average squared distance between points and centroids

**Important Properties**:
- Always converges to a local minimum (not necessarily global)
- Results depend on initial centroids (may need multiple runs)
- K is a hyperparameter (must be chosen beforehand)
- Tends to find spherical clusters of similar sizes
- Sensitive to outliers

**Soft K-Means Clustering**:
- Assigns probabilistic memberships instead of hard assignments
- rn,k = e^(-β‖xn-μk‖²)/sum(e^(-β‖xn-μj‖²))
- β controls "softness" of assignments (higher β → harder boundaries)
- Converges to standard K-means as β → ∞

### 3. K-Means Example

**Example**: D = {1, 2, 3, 10, 17, 20}, K=2, initial μ1=0, μ2=5

**Iteration 1**:
- Distances from μ1=0: |1-0|=1, |2-0|=2, |3-0|=3, |10-0|=10, |17-0|=17, |20-0|=20
- Distances from μ2=5: |1-5|=4, |2-5|=3, |3-5|=2, |10-5|=5, |17-5|=12, |20-5|=15
- Assignments: C1={1,2,3}, C2={10,17,20}
- New centroids: μ1=(1+2+3)/3=2, μ2=(10+17+20)/3=15.67

**Iteration 2**:
- Distances from μ1=2: |1-2|=1, |2-2|=0, |3-2|=1, |10-2|=8, |17-2|=15, |20-2|=18
- Distances from μ2=15.67: |1-15.67|=14.67, |2-15.67|=13.67, |3-15.67|=12.67, |10-15.67|=5.67, |17-15.67|=1.33, |20-15.67|=4.33
- Assignments: C1={1,2,3}, C2={10,17,20} (unchanged)
- New centroids: μ1=2, μ2=15.67 (unchanged)

**Final Result**: Converged with clusters C1={1,2,3}, C2={10,17,20}

### 4. Probabilistic View of Clustering

**Gaussian Mixture Model Connection**:
- K-means is equivalent to fitting data with K Gaussians where:
  * Each Gaussian has the same spherical covariance matrix (σ²I)
  * Each point is assigned to the most likely Gaussian
  * Only the means μk are estimated

**K-means as Iterative Maximum Likelihood**:
- **E-step** (Cluster Assignment): Find most likely cluster for each point
- **M-step** (Centroid Update): Update means using maximum likelihood

**K-means Limitations**:
- Can get stuck in local minima (try multiple initializations)
- Assumes clusters are spherical and equally sized
- Sensitive to outliers
- K must be specified in advance

### 5. Testing/Validation

**Evaluating Model Performance**:
- Split data into Training, Validation, Test sets
- Training: Learn model parameters
- Validation: Tune hyperparameters (e.g., K in K-means)
- Testing: Evaluate final performance

**Checking Generalization**:
- JTest ≈ JTrain → model generalizes well
- JTest >> JTrain → model overfits
- For K-means: Test if risk (distortion) is similar on test data

**Choosing K** (number of clusters):
- Try different values and look for "elbow" in distortion plot
- Use domain knowledge about expected number of groups
- Consider silhouette score or other cluster validity indices
- Cross-validation can help determine optimal K