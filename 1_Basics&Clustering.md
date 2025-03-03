# **Unit 1: Machine Learning Basics & Clustering**

---

## **1. General Idea**
### **What problem are we solving?**
- **Goal:** Understand the **basic concepts of machine learning** and how we categorize learning problems.
- **Key Idea:** Machine learning can be divided into:
  - **Supervised Learning** (we have labeled data, e.g., classification & regression).
  - **Unsupervised Learning** (we only have data, no labels, e.g., clustering).
- **K-Means Clustering**: A widely used method for **grouping data** into clusters.

---

## **2. Definitions**
These are the key terms and definitions you must know.

| **Term**                | **Definition** |
|------------------------|--------------|
| **Dataset**            | A set of collected samples (e.g., customer purchases, sensor readings). |
| **Model**              | A function \( f(x) \) that represents the **pattern or structure** in the data. |
| **Algorithm**          | A **procedure** that uses the dataset to find the **best model parameters**. |
| **Supervised Learning** | Learning from **labeled** data to predict outputs (e.g., classification, regression). |
| **Unsupervised Learning** | Learning from **unlabeled** data to find patterns (e.g., clustering, dimensionality reduction). |
| **Empirical Risk**      | The **average loss** over a dataset. In general: \( R(f) = \frac{1}{N} \sum L(y_n, v_n) \). |
| **Loss Function**       | A function that measures **how wrong** a model’s prediction is (e.g., MSE, cross-entropy). |
| **K-Means Algorithm**   | A method to partition data into **\( K \) clusters**, each represented by a centroid. |

---

## **3. Solution Process**
This section gives a **clear, step-by-step formula derivation** for clustering.

### **Step 1: Define the K-Means Algorithm**
The K-Means algorithm follows an **iterative process**:
1. **Initialize \( K \) centroids** randomly.
2. **Assign each data point** to the **nearest** centroid.
3. **Update each centroid** to be the mean of points assigned to it.
4. Repeat **until convergence** (centroids stop changing).

### **Step 2: Clustering Objective Function**
- We aim to minimize the **sum of squared distances** between points and their assigned centroids.
- The **distortion function (cost function)** is:

  \[
  J = \sum_{n=1}^{N} \sum_{k=1}^{K} r_{n,k} || x_n - \mu_k ||^2
  \]

  where:
  - \( r_{n,k} \) is 1 if \( x_n \) belongs to cluster \( k \), otherwise 0.
  - \( \mu_k \) is the centroid of cluster \( k \).

### **Step 3: Clustering Update Steps**
1. **Cluster Assignment Step**  
   Assign each data point to the closest centroid:

   \[
   r_{n,k} = 1 \quad \text{if} \quad k = \arg \min_j || x_n - \mu_j ||^2
   \]

2. **Centroid Update Step**  
   Compute new centroids as the mean of assigned points:

   \[
   \mu_k = \frac{1}{|C_k|} \sum_{x_n \in C_k} x_n
   \]

---

## **4. Sample Numerical Example**
### **Problem Statement:**
We have a dataset:  
\[
D = \{1, 2, 3, 10, 17, 20\}
\]
We want to cluster it into **\( K = 2 \) groups** using **K-Means**.

### **Step 1: Initialize Centroids**
Let’s start with:
\[
\mu_1 = 0, \quad \mu_2 = 5
\]

### **Step 2: Assign Points to the Closest Centroid**
| Data Point | Distance to \( \mu_1 = 0 \) | Distance to \( \mu_2 = 5 \) | Assigned Cluster |
|------------|----------------------|----------------------|-----------------|
| 1          | 1                    | 4                    | 1 |
| 2          | 2                    | 3                    | 1 |
| 3          | 3                    | 2                    | 2 |
| 10         | 10                   | 5                    | 2 |
| 17         | 17                   | 12                   | 2 |
| 20         | 20                   | 15                   | 2 |

New clusters:
- **Cluster 1**: {1, 2}
- **Cluster 2**: {3, 10, 17, 20}

### **Step 3: Compute New Centroids**
\[
\mu_1 = \frac{1+2}{2} = 1.5, \quad \mu_2 = \frac{3+10+17+20}{4} = 12.5
\]

### **Step 4: Repeat Until Convergence**
1. **Reassign clusters** based on new centroids.
2. **Recalculate centroids**.
3. Stop when centroids **no longer change**.

**Final Clusters:**
\[
C_1 = \{1, 2, 3\}, \quad C_2 = \{10, 17, 20\}
\]
\[
\mu_1 = 2, \quad \mu_2 = 15.67
\]

✔ **Clustering is done!**

---

## **5. Other Important Details**
Here are some additional details that are **not explicitly needed for solving the midterm problems** but are still important for a **deeper understanding**.

- **Soft K-Means (Probabilistic Clustering)**  
  Instead of **hard assigning** each point to a single cluster, **Soft K-Means** assigns **probabilities** of belonging to each cluster using a function:
  \[
  r_{n,k} = \frac{e^{-\beta ||x_n - \mu_k||^2}}{\sum_{j=1}^{K} e^{-\beta ||x_n - \mu_j||^2}}
  \]
  where **\( \beta \)** controls how "soft" the assignments are.

- **Choosing \( K \) in K-Means**  
  - The **Elbow Method**:  
    - Compute clustering cost for different \( K \).
    - Plot cost vs. \( K \) and find the "elbow" (point where adding more clusters **does not reduce cost significantly**).
  - **Silhouette Score**: Measures how well points fit into their assigned clusters.

- **K-Means Limitations**  
  - **Assumes spherical clusters** (doesn’t work well with elongated or overlapping clusters).
  - **Sensitive to initialization** (bad initial centroids can lead to poor clusters).
  - **May converge to local minima** (use multiple runs to find best clustering).