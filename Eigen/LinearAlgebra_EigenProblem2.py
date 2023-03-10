#!/usr/bin/env python
# coding: utf-8

# # Numerical Solution of All Eigenvalues

# In this section we'll continue our discussion on solving for eigenvalues.  First, we introduce the Householder similarity transform as a method to tridiagonalize a symmetric matrix without changing its eignenvalues.  Then we discuss an efficient and simple deflation technique to solve for all the eigenvalues of a matrix.

# ## Householder's Method

# Householder's Method is a similarity transform. We will use it to find the eigenvalues of a matrix, but it has other uses outside this process. The method is used to find a symmetric tridiagonal matrix $\mathbf{B}$ which is similar to a given symmetric matrix $\mathbf{A}$. We'll start by defining the *Householder Transformation*

# Let $\vec{\omega} \in \mathbb{R}^n$ with $\vec{\omega}\vec{\omega}^\intercal = 1$. The $n\times n$ matrix,

# #### Householder transformation:

# $$\mathbf{P} = \mathbf{I}-2 \vec{\omega}\vec{\omega}^\intercal$$

# $$\mbox{with}\quad \vec\omega \vec{\omega}^\intercal = 1$$

# This transformation is symmetric and orthogonal, therefore $\mathbf{P}^{-1} = \mathbf{P}$.
# 
# Householder's method begins by determining a transform $\mathbf{P}^{(1)}$ with the property, $\mathbf{A}^{(2)} = \mathbf{P}^{(1)} \mathbf{A} \mathbf{P}^{(1)}$ that has

# Determine
# 
# $$\mathbf{A}^{(2)} = \mathbf{P}^{(1)} \mathbf{A} \mathbf{P}^{(1)}$$

# $$a_{j1}^{(2)} = 0, \quad \forall \quad j=3, 4, \ldots, n$$

# and by symmetry, $a_{1j}^{(2)} = 0$. The desired outcome of this transformation along with the fact that $\vec{\omega}\vec{\omega}^\intercal = 1$ by definition, imposes $n$ constraints on the $n$ unknown components of $\vec{\omega}$. If we set $\omega_1 = 0$, then $a_{11}^{(2)} = a_{11}$. Now, we want
# 
# $$\mathbf{P}^{(1)} = \mathbf{I} - 2\vec{\omega}\vec{\omega}^\intercal$$
# 
# to satisfy
# 
# $$\mathbf{P}^{(1)}\left[a_{11}, a_{21}, a_{31}, \ldots a_{n1}\right]^\intercal = \left[a_{11}, \alpha, 0, \ldots, 0\right]^\intercal, \label{eqn:Ptrans1}\tag{1}$$
# 
# where $\alpha$ will be chosen later. Defining some notation
# 
# $$\hat{\vec{\omega}} = \left[\omega_2, \omega_3, \ldots, \omega_n\right]^\intercal \in\mathbb{R}^{n-1},\\ \hat{\vec{y}} = \left[a_{21}, a_{31}, \ldots, a_{n1}\right]^\intercal \in\mathbb{R}^{n-1},$$
# 
# and $\hat{\mathbf{P}}$ is the $(n-1) \times (n-1)$ Householder transformation
# 
# $$\hat{\mathbf{P}} = \mathbf{I}_{n-1} - 2\hat{\vec{\omega}}\hat{\vec{\omega}}^\intercal.$$
# 
# with these definitions, (\ref{eqn:Ptrans1}) becomes
# 
# $$\hat{\mathbf{P}}^{(1)} \begin{bmatrix}a_{11} \\a_{21} \\\vdots \\a_{n1} \end{bmatrix}= \begin{bmatrix} 1 & \vec{0}^\intercal \\ \vec{0} & \hat{\mathbf{P}} \end{bmatrix}  \begin{bmatrix} a_{11} \\ \hat{\vec{y}} \end{bmatrix} = \begin{bmatrix}a_{11} \\ \hat{\mathbf{P}} \hat{\vec{y}} \end{bmatrix} = \begin{bmatrix} a_{11} \\ \alpha \\ \vec{0} \end{bmatrix}$$
# 
# with
# 
# $$\hat{\mathbf{P}}\hat{\vec{y}} = \left[\mathbf{I}_{n-1} - 2\vec{\omega} \vec{\omega}^\intercal\right]\hat{\vec{y}} = \hat{\vec{y}} - 2(\hat{\vec{\omega}}^\intercal\hat{\vec{y}})\hat{\vec{\omega}} = \left[\alpha ,0,\ldots,0\right]^\intercal  \label{eqn:Ptrans2}\tag{2}$$
# 
# Let $r = \hat{\vec{\omega}}^\intercal \hat{\vec{y}}$. Then,
# 
# $$\left[\alpha, 0, \ldots, 0\right]^\intercal = \left[a_{21} - 2 r \omega_2, a_{31}-2r \omega_3, \ldots, a_{n1} - 2r \omega_n\right]^\intercal$$
# 
# we can determine all of the $\omega_j$ once we know $\alpha$ and $r$. Equating components
# 
# $$\alpha = a_{21} - 2r \omega_2$$
# 
# and
# 
# $$0 = a_{j1} - 2r \omega_j \quad \mathrm{for} \quad j=3,4,\ldots,n$$
# 
# Rearranging slightly,
# 
# $$2r \omega_2 = a_{21} - \alpha \tag{3}\label{eqn:2romega}$$
# 
# $$2r \omega_j = a_{j1} \quad \mathrm{for} \quad j=3,4,\ldots,n \tag{4}\label{eqn:omegaj}$$
# 
# Now we square both sides of each equation and add the equations together,
# 
# $$4 r^2 \sum_{j=2}^n \omega_j^2 = (a_{21} - \alpha)^2 + \sum_{j=3}^n a_{j1}^2.$$
# 
# Since $\hat{\vec{\omega}}\hat{\vec{\omega}}^\intercal = 1$ and $\omega_1 = 0$, we have $\sum_{j=2}^n \omega_j^n = 1$, and
# 
# $$4r^2 = \sum_{j=2}^n a_{j1}^2 - 2 \alpha a_{21} + \alpha^2. \tag{5}\label{eqn:4r2}$$
# 
# Equation (\ref{eqn:Ptrans2}) and the fact that $\mathbf{P}$ is orthogonal implies that
# 
# $$\alpha ^2 = \left[\alpha, 0, \ldots, 0\right]\left[\alpha, 0, \ldots, 0\right]^\intercal = \left[\hat{\mathbf{P}} \hat{\vec{y}}\right]^\intercal \hat{\mathbf{P}}\hat{\vec{y}} = \hat{\vec{y}}\hat{\mathbf{P}}^\intercal\hat{\mathbf{P}}\hat{\vec{y}} = \hat{\vec{y}}^T\hat{\vec{y}}$$
# 
# Thus,
# 
# $$\alpha^2 = \sum_{j=2}^n a_{j1}^2,$$
# 
# Substituting into (\ref{eqn:4r2})
# 
# $$2r^2 = \sum_{j=2}^n a_{j1}^2 - \alpha a_{21}$$
# 
# To ensure $2r^2 = 0$ only when $a_{21} = a_{31} = \ldots = a_{n1} = 0$, we chose
# 
# $$\alpha = -\mathrm{sgn}\left(a_{21}\right)\sqrt{\sum_{j=2}^n a_{j1}^2}$$
# 
# therefore,
# 
# $$2r^2 = \sum_{j=2}^n a_{j1}^2 + \vert a_{21}\vert\sqrt{\sum_{j=2}^n a_{j1}^n}$$
# 
# With this choice of $\alpha$ and $2r^2$ we can solve (\ref{eqn:2romega}) and (\ref{eqn:omegaj}) to obtain,
# 
# $\omega_2 = \frac{a_{21} - \alpha}{2r}$ and  $\omega_j = \frac{a_{j1}}{2r}$, and for $j = 3, \ldots, n$
# 
# To summarize:
# 
# $$
# \alpha = -\mathrm{sgn}\left(a_{21}\right)\sqrt{\sum_{j=2}^n a_{j1}^n}\\
# r = \sqrt{\frac{1}{2} \alpha^2 - \frac{1}{2}a_{21}\alpha}\\
# \omega_1 = 0\\
# \omega_2 = \frac{a_{21}-\alpha}{2r}\\
# \omega_j = \frac{a_{j1}}{2r}, \quad \mbox{for each} \quad j=3, \ldots, n$$
# 
# 
# For these choices we have:

# $$\mathbf{A}^{(2)} = \mathbf{P}^{(1)}\mathbf{AP}^{(1)} = \begin{bmatrix} a_{11}^{(2)} & a_{12}^{(2)} & 0 & \ldots & 0\\a_{21}^{(2)} & a_{22}^{(2)} & a_{23}^{(2)} & \ldots & a_{2n}^{(2)} \\ 0 & a_{32}^{(2)} & a_{33}^{(2)} & \ldots & a_{3n}^{(2)} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & a_{n2}^{(2)} & a_{n3}^{(2)} & \ldots & a_{nn}^{(2)} \end{bmatrix}$$

# Having found $\mathbf{P}^{(1)}$ and computed $\mathbf{A}^{(2)}$, the process is repeated for $k = 2, 3, \ldots, n-2$:
# 
# $$\alpha = -\mathrm{sgn}\left(a_{k+1,k}^{(k)}\right) \sqrt{\sum_{j=2}^n (a_{jk}^{(k)})^2}\\
# r = \sqrt{\frac{1}{2}\alpha^2 - \frac{1}{2} \alpha \alpha_{k+1,k}^{(k)}}\\
# \omega_1^{(k)} = \omega_2^{(k)} = \ldots = \omega_k^{(k)} = 0\\
# \omega_{k+1}^{(k)} = \frac{a_{k+1,k}^{(k)} -\alpha}{2r}$$
# 
# $\omega_j^{(k)} = \frac{a_{jk}}{2r}$, for each $j = k+2, k+3, \ldots, n$

# Continuing

# $$\mathbf{P}^{(k)} = \mathbf{I} -2\vec{\omega}^{(k)} \cdot \vec{\omega}^{(k)\intercal}$$
# 
# $$\mathbf{A}^{(k+1)} = \mathbf{P}^{(k)}\mathbf{A}^{(k)}\mathbf{P}^{(k)}$$

# Continuing in this manner, the tridiagonal and symmetric matrix $\mathbf{A}^{(n-1)}$ is formed
# 
# $$\mathbf{A}^{(n-1)} = \mathbf{P}^{(n-2)}\mathbf{P}^{(n-3)}\ldots \mathbf{P}^{(1)}\mathbf{AP}^{(1)}\ldots \mathbf{P}^{(n-3)}\mathbf{P}^{(n-2)}.$$

# ### Psuedocode for Householder's Method
# 
# Give a symmetric matrix $\mathbf{A}$ with $n$ number of rows
# 
# | Steps | |
# | --: | :-- |
# | 1.  | For $k = 1, 2, \ldots, n-2$ do steps 2, 5-8, 10, 12, 14, 19-20, 22-23
# | 2.  | $\phantom{--}$ If $a_{k+1, k} = 0$ do Step 3.
# | 3.  | $\phantom{----}$ Set $$\alpha = -\sqrt{\sum_{j=k+1}^n (a_{jk})^2}$$
# |     | $\phantom{--}$ Else, do Step 4.
# | 4.  | $\phantom{----}$ Set $$\alpha = -\mathrm{sgn}(a_{k+1, k})\sqrt{\sum_{j=k+1}^n(a_{jk})^2}$$
# | 5.  | $\phantom{--}$ Set $2r^2 = \alpha^2 - \alpha a_{k+1,k}$
# | 6.  | $\phantom{--}$ Set $v_k = 0$
# | 7.  | $\phantom{--}$ Set $v_{k+1} = a_{k+1,k} - \alpha$
# | 8.  | $\phantom{--}$ For $j = k + 2, \ldots , n$ do Step 9
# | 9.  | $\phantom{----} v_j = a_{jk}$ (note $\vec{\omega} = \frac{1}{2r} \vec{v}$)
# | 10. | $\phantom{--}$ For $j = k, k+1, \ldots, n$ do Step 11
# | 11. | $\phantom{----} u_j = \frac{1}{2 r^2} \sum_{i = k+1}^n a_{ji}v_i$ (note $\vec{u} = \frac{1}{r}A\vec{\omega}$)
# | 12. | $\phantom{--}$ For $j = k, k+1, \ldots, n$ do Step 13
# | 13. | $\phantom{----} z_j = u_j - \left(\frac{\vec{u}\cdot\vec{v}}{4 r^2}\right)v_j$  (note $\vec{z} = \frac{1}{r}\mathbf{A}\vec{\omega} - \vec{\omega}\vec{\omega}^\intercal \frac{1}{r} \mathbf{A}\vec{\omega}$)
# | 14. | $\phantom{--}$ For $l = k+1, k+2, \ldots ,n-1$ do Steps 15, 18
# | 15. | $\phantom{----}$ For $j = l + 1, \ldots, n$ Do Steps 16-17
# | 16. | $\phantom{------} a_{jl} = a_{jl} - v_l z_j - v_j z_l$
# | 17. | $\phantom{------} a_{lj} = a_{jl}$
# | 18. | $\phantom{----}$ Set $a_{ll} = a_{ll} - 2v_l z_l$
# | 19. | $\phantom{--}$ Set $a_{nn} = a_{nn} - 2v_n z_n$
# | 20. | $\phantom{--}$ For $j=k+2, \ldots, n$ do Step 21
# | 21. | $\phantom{----}$ Set $a_{kj} = a_{jk} = 0$
# | 22. | $\phantom{--}$ Set $a_{k+1, k} = a_{k+1, k} - v_{k+1}z_k$
# | 23. | $\phantom{--}$ Set $a_{k, k+1} = a_{k+1, k}$

# ### Python/NumPy implementation of Householder's method

# In[7]:


import numpy as np


# In[8]:


def householder(A):

    n = A.shape[0]
    v = np.zeros(n, dtype=np.double)
    u = np.zeros(n, dtype=np.double)
    z = np.zeros(n, dtype=np.double)

    for k in range(0, n - 2):

        if np.isclose(A[k + 1, k], 0.0):
            ?? = -np.sqrt(np.sum(A[(k + 1) :, k] ** 2))
        else:
            ?? = -np.sign(A[k + 1, k]) * np.sqrt(np.sum(A[(k + 1) :, k] ** 2))

        two_r_squared = ?? ** 2 - ?? * A[k + 1, k]
        v[k] = 0.0
        v[k + 1] = A[k + 1, k] - ??
        v[(k + 2) :] = A[(k + 2) :, k]
        u[k:] = 1.0 / two_r_squared * np.dot(A[k:, (k + 1) :], v[(k + 1) :])
        z[k:] = u[k:] - np.dot(u, v) / (2.0 * two_r_squared) * v[k:]

        for l in range(k + 1, n - 1):

            A[(l + 1) :, l] = (
                A[(l + 1) :, l] - v[l] * z[(l + 1) :] - v[(l + 1) :] * z[l]
            )
            A[l, (l + 1) :] = A[(l + 1) :, l]
            A[l, l] = A[l, l] - 2 * v[l] * z[l]

        A[-1, -1] = A[-1, -1] - 2 * v[-1] * z[-1]
        A[k, (k + 2) :] = 0.0
        A[(k + 2) :, k] = 0.0

        A[k + 1, k] = A[k + 1, k] - v[k + 1] * z[k]
        A[k, k + 1] = A[k + 1, k]


# ## Gram-Schmidt Process

# In an *orthogonal basis* of a vector space (matrix), every vector is perpendicular to every other vector. If we divide each vector by its length we have unit vectors that span the vector space which we will call an *orthonormal basis*. A common orthonormal basis is the *standard basis* represented as the identity matrix. This is not the only orthonormal basis, we can rotate the axes without changing the right angles at which the vectors meet. These rotation matrices, we will call $\mathbf{Q}$. Every basis has an orthonormal basis which can be constructed by a simple process known as *Gram-Schmidt orthonormalization*. This process takes a skewed set of axes and makes them perpendicular.
# 
# In numerical linear algebra orthonormal matrices are important because they don't introduce any numerical instability. When lengths stay the same, roundoff error stays in control. Suppose we are given a $3\times 3$ matrix $\mathbf{A} = \left[\vec{a} \vert \vec{b} \vert \vec{c}\right]$, where $\vec{a}, \vec{b}, \vec{c}$ are the column vectors of $\mathbf{A}$. We want to produce three column vectors, $\vec{q}_1, \vec{q}_2$, and $\vec{q}_3$ which form an orthonormal basis of $\mathbf{A}$. To start we can choose $\vec{q}_1$ to be in the same direction of $\mathbf{A}$ for convenience, all we have to do is make it a unit vector.

# Consider the $3 \times 3$ matrix $\mathbf{A}$
# $$\mathbf{A} = \left[\vec{a} \vert \vec{b} \vert \vec{c}\right]$$

# $$\vec{q}_1 = \frac{\vec{a}}{\vert a \vert}$$

# For $\vec{q}_2$ we must subtract any component of $\vec{b}$ that is in the direction of $\vec{q}_1$. We calculate the projection of $\vec{b}$ onto $\vec{q}_1$ by $(\vec{q}_1^\intercal \vec{b})\vec{q}_1$, subtracting for this projection from $\vec{b}$, gives us an intermediate orthogonal vector $\vec{B}$, that we can make orthonormal by dividing by its length:

# $$\vec{B} = \vec{b}- \left(\vec{q}_1^\intercal \vec{b}\right)\vec{q}_1$$

# $$\vec{q}_2 = \frac{\vec{B}}{\vert \vec{B}\vert}$$

# To get $\vec{c}$ to $\vec{q}_3$ we need to subtract any components of $\vec{c}$, that might lie in the $\vec{q}_1, \vec{q}_2$ plane. What is left over is an orthogonal vector to $\vec{q}_1$ and $\vec{q}_2$, $\vec{C}$, that we make orthonormal by dividing by its length

# $$\vec{C} = \vec{c}-\left(q_1^\intercal \vec{c}\right)\vec{q}_1 - \left(\vec{q}_2^\intercal \vec{c}\right)\vec{q}_2$$
# 
# $$\vec{q}_3 = \frac{\vec{C}}{\vert C \vert}$$

# The matrix $\mathbf{Q}$, then forms an orthonormal basis of $\mathbf{A}$.

# $$\mathbf{Q} = \left[\vec{q}_1 \vert \vec{q}_2 \vert \vec{q}_3  \right]$$

# ### Pseudocode for Gram-Schmidt
# 
# Given a matrix $\mathbf{A} = \begin{bmatrix} \vec{x}_1 \vert \vec{x}_2 \vert \ldots \vert \vec{x}_n \end{bmatrix}$
# 
# | Steps |     |
# | --:   | :-- |
# | 1.    | For  $j=1, 2, \ldots, n$ do Step 2-3
# | 2.    | $\phantom{--}$ Set  $\vec{v}_j=\vec{x}_j$
# | 3.    | For  $j=1, 2, \ldots, n$ do Step 4-5
# | 4.    | $\phantom{--}$ Set  $\vec{q}_j=\vec{v}_j / \vert \vec{v}_j \vert$
# |???5.    | $\phantom{--}$ For  $k=j+1, 2, \ldots, n$ do Step 6
# | 6.    |???$$\phantom{----} \vec{v}_k=\vec{v}_k-\left(\vec{q}^\intercal_k \vec{v}_j\right)\vec{q}_k$$

# ### Python/NumPy implementation of Gram-Schmidt

# In[9]:


def gram_schmidt(A):

    m = A.shape[1]
    Q = np.zeros(A.shape, dtype=np.double)
    temp_vector = np.zeros(m, dtype=np.double)

    Q[:, 0] = A[:, 0] / np.linalg.norm(A[:, 0], ord=2)

    for i in range(1, m):
        q = Q[:, :i]
        temp_vector = np.sum(np.sum(q * A[:, i, None], axis=0) * q, axis=1)
        Q[:, i] = A[:, i] - temp_vector
        Q[:, i] /= np.linalg.norm(Q[:, i], ord=2)

    return Q


# ## QR Factorization

# What is the relationship between the matrices $\mathbf{A}$ and $\mathbf{Q}$ from the Gram-Schmidt process? The relationship between $\vec{a}$ and $\vec{q}_1$ is trivial since we set them equal to one another. The vector $\vec{b}$ is a combination of the orthonormal $\vec{q}_1$ and $\vec{q}_2$, and we know what that relationship is

# $$\vec{b} = \left(\vec{q}_1^T \vec{b}\right)\vec{q}_1 + \left(\vec{q}_2^\intercal \vec{b}\right)\vec{q}_2$$

# Similarly $\vec{c}$ is the sum of its $\vec{q}_i$'s

# $$\vec{c} = \left(q_1^T \vec{c}\right)\vec{q}_1 + \left(\vec{q}_2^\intercal \vec{c}\right)\vec{q}_2 + \left(\vec{q}_3^\intercal \vec{c}\right)\vec{q}_3$$

# If we write these relationships in matrix form, we have

# $$
# \mathbf{A} = \left[\vec{a} \vert \vec{b} \vert \vec{c}\right] = \left[\vec{q}_1 \vert \vec{q}_2 \vert \vec{q}_3\right]
# \begin{bmatrix}\left(\vec{q}_1^\intercal \vec{a}\right) & \left(\vec{q}_1^\intercal \vec{b}\right) & \left(\vec{q}_1^\intercal \vec{c}\right) \\
# 0 & \left(\vec{q}_2^\intercal \vec{b}\right) & \left(\vec{q}_2^\intercal \vec{c}\right) \\
# 0 & 0 & \left(\vec{q}_3^\intercal \vec{c}\right)
# \end{bmatrix} = \mathbf{QR}
# $$

# ## QR Algorithm for computing eigenvalues

# This algorithm is so simple it's scary. We start with a matrix $\mathbf{A}^{(0)}$ and perform $\mathbf{QR}$ factorization,

# $$\mathbf{A}^{(0)} = \mathbf{Q}^{(0)}\mathbf{R}^{(0)}$$

# Then we reverse the factors to form the matrix $\mathbf{A}^{(1)}$,

# $$\mathbf{A}^{(1)} = \mathbf{R}^{(0)}\mathbf{Q}^{(0)}$$

# $$\mathbf{A}^{(1)} = \mathbf{R}^{(0)}\mathbf{Q}^{(0)} \tag{6}\label{eqn:reverseQR}$$

# We can do this because $\mathbf{A}^{(1)}$ is similar to $\mathbf{A}^{(0)}$, we can verify
# 
# $$\mathbf{Q}^{(0)^{-1}}\mathbf{A}^{(0)}\mathbf{Q}^{(0)} = \mathbf{Q}^{(0)^{-1}}(\mathbf{Q}^{(0)}\mathbf{R}^{(0)})\mathbf{Q}^{(0)} = \mathbf{A}^{(1)}$$

# And we continue this process without changing the eigenvalues. The above process is called the *unshifted QR algorithm*, and almost always $\mathbf{A}^{(k)}$ approaches a triangular form, its diagonal entries approach its eigenvalues, which are also the eigenalues of $\mathbf{A}^{(0)}$. This algorithm is okay, but we can make it even better by using two refinements

# ### Shifted $\mathbf{QR}$
# 
# $$\mathbf{A}^{(k)} - \alpha^{(k)}\mathbf{I} = \mathbf{Q}^{(k)}\mathbf{R}^{(k)}$$

# $$\mathbf{A}^{(k+1)} = \mathbf{R}^{(k)}\mathbf{Q}^{(k)} + \alpha^{(k)}\mathbf{I}$$

# $$\mathbf{A}^{(k)} = \begin{bmatrix} * & * & * & * \\ * & * & * & * \\ 0 & * & * & * \\ 0 & 0 & \epsilon & \mu_1 \end{bmatrix}, \quad \mathrm{where} \quad \epsilon<< 1.$$

#   1. We allow shifts, in a similar fashion to the inverse power method. If the number $\alpha^{(k)}$ is close to an eigenvalue, the step in (\ref{eqn:reverseQR}) should be shifted as follows
# 
#      $\mathbf{A}^{(k)} - \alpha^{(k)}\mathbf{I} = \mathbf{Q}^{(k)}\mathbf{R}^{(k)}$ and then $\mathbf{A}^{(k+1)} = \mathbf{R}^{(k)}\mathbf{Q}^{(k)} + \alpha^{(k)}\mathbf{I}$
# 
#      One popular method of choosing the shifted constant, $\alpha^{(k)}$ is to simply use the $a_{nn}^{(k)}$ entry of the matrix. Another method uses a shifting constant that is the eigenvalue of the matrix
# 
#     $$\begin{bmatrix} a_{n-1,n-1}^{(k)} & a_{n-1,n}^{(k)} \\ a_{n,n-1}^{(k)} & a_{nn}^{(k)}\end{bmatrix}$$
# 
#     that is closest to $a_{nn}^{(k)}$.
# 
#     After a few iterations the matrix $\mathbf{A}^{(k)}$ will come to look as follows:
# 
#     $$\mathbf{A}^{(k)} = \begin{bmatrix} * & * & * & * \\ * & * & * & * \\ 0 & * & * & * \\ 0 & 0 & \epsilon & \mu_1 \end{bmatrix}, \quad \mathrm{where} \quad \epsilon<< 1.$$
# 
#     $\mu_1$ will be accepted as $\lambda_1$. We will find the next eigenvalue by eliminating the last row and column and performing the same procedure with the smaller submatrix. And continue until all eigenvalues have been found. At this point the $\mathbf{QR}$ algorithm is completely described. If the eigenvectors are desired they are one step away with an inverse power method using the eigenvalues found here.
# 
#   2. The last thing we can do to greatly improve the $\mathbf{QR}$ algorithm is to make the $\mathbf{QR}$ factorization very fast at each step. This is where the Housholder transformation comes in. It turns out that the $\mathbf{QR}$ factorization generally takes $O\left(n^3\right)$ operations, however for a matrix that is already in tridiagonal form this becomes $O\left(n\right)$ and once the matrix $\mathbf{A}^{(0)}$ is in tridiagonal form, each $\mathbf{A}^{(k)}$ will stay that way at each step.

# ### Pseudocode for $\mathbf{QR}$ Algorithm for computing eigenvalues
# 
# Starting with a symmetric tridiagonal matrix, we will relabel some of the terms for simplicity:
# 
# $$A = \begin{bmatrix} a_1 & b_2 & 0 & \ldots & 0 \\ b_2 & a_2 & b_3 & \ddots & \vdots \\ 0 & b_3 & \ddots & \ddots & 0 \\ \vdots & \ddots & \ddots & \ddots & b_n \\ 0 & \ldots & 0 & b_n & a_n \end{bmatrix}$$
# 
# | Steps | |
# | --: | :-- |
# | 1.  | Set $k = 1$,  Set $\mathtt{SHIFT} = 0$
# | 2.  | While $k \leq M$ do Steps 3,6,13,15,18,22-23,32-34,36-37,46-49,52-54
# | 3.  | $\phantom{--}$ If $\vert b_n \vert \leq TOL$ do Steps 4-5
# | 4.  | $\phantom{----}$ Set $\lambda_n = a_n + \mathtt{SHIFT}$
# | 5.  | $\phantom{----}$ Set $n = n-1$
# | 6.  | $\phantom{--}$ If $\vert b_2\vert \leq TOL$ do Steps 7-10
# | 7.  | $\phantom{----}$ Set $\lambda_2 = a_1 + \mathtt{SHIFT}$
# | 8.  | $\phantom{----}$ Set $n = n-1$
# | 9.  | $\phantom{----}$ Set $a_1 = a_2$
# | 10. | $\phantom{----}$ For $j = 2, \ldots, n$ do Steps 11-12
# | 11. | $\phantom{------}$ Set $a_j = a_{j+1}$
# | 12. | $\phantom{------}$ Set $b_j = b_{j+1}$
# | 13. | $\phantom{--}$ If $n = 0$ do Step 14
# | 14. | $\phantom{----}$ Break out of loop.
# | 15. | $\phantom{--}$ If $n = 1$ then
# | 16. | $\phantom{----}$ Set $\lambda_1 = a_1 + \mathtt{SHIFT}$
# | 17. | $\phantom{----}$ Break out of loop.
# | 18. | $\phantom{--}$ For $j = 3, \ldots, n-1$ do Step 19
# | 19. | $\phantom{----}$ If $\vert b_j\vert  \leq TOL$ do Step 20-21
# | 20. | $\phantom{------}$ Print "Split into $a_1, \ldots, a_{j-1}, b_2, \ldots, b_{j-1}$ and $a_j, \ldots a_n, b_{j+1}, \ldots, b_n$"
# | 21. | $\phantom{------}$ Break out of loop.
# | 22. | $\phantom{--}$ Set $B = -(a_{n-1} + a_n), C = a_n a_{n-1} - b_n^2, D = \sqrt{B^2 -4C}$
# | 23. | $\phantom{--}$ If $B > 0$ do Steps 25-26
# | 24. | $\phantom{----}$ Set $\mu_1 = -2C/(B+D)$
# | 25. | $\phantom{----}$ Set $\mu_2 = -(B+D)/2$
# |     | $\phantom{--}$ Else do Steps 26-27
# | 26. | $\phantom{----}$ Set $\mu_1 = (D-B)/2$
# | 27. | $\phantom{----}$ Set $\mu_2 = 2C/(D-B)$
# | 28. | $\phantom{--}$ If $n = 2$ then do Steps 29-31
# | 29. | $\phantom{----}$ Set $\lambda_1 = \mu_1 + \mathtt{SHIFT}$
# | 30. | $\phantom{----}$ Set $\lambda_2 = \mu_2 + \mathtt{SHIFT}$
# | 31. | $\phantom{----}$ Break out of loop.
# | 32. | $\phantom{--}$ Choose $S$ so that $\vert S-a_n \vert = \min\left(\vert \mu_1-a\vert, \vert \mu_2-a_2 \vert \right)$.
# | 33. | $\phantom{--}$ Set $\mathtt{SHIFT} = \mathtt{SHIFT}+S$
# | 34. | $\phantom{--}$ For $j = 1, \ldots, n$ do Step 35
# | 35. | $\phantom{----}$ Set $d_j = a_j -S$
# | 36. | $\phantom{--}$ Set $x_1 = d_1, y_1 = b_2$
# | 37. | $\phantom{--}$ For $j = 2, \ldots, n$ do Steps 38-43
# | 38. | $\phantom{----}$ Set $z_{j-1} = (x_{j-1}^2 + b_j^2)^{1/2}$
# | 39. | $\phantom{----}$ Set $c_j = x_{j-1}/z_{j-1}$
# | 40. | $\phantom{----}$ Set $s_j = b_j / z_{j-1}$
# | 41. | $\phantom{----}$ Set $q_{j-1} = c_j y_{j-1} + s_j d_j$
# | 42. | $\phantom{----}$ Set $x_j = -s_jy_{j-1} + c_jd_j$
# | 43. | $\phantom{----}$ If $j \neq n$ do Steps 44-45
# | 44. | $\phantom{------}$ Set $r_{j-1} = s_j b_{j+1}$
# | 45. | $\phantom{------}$ Set $y_j = c_j b_{j+1}$
# | 46. | $\phantom{--}$ Set $z_n = x_n$
# | 47. | $\phantom{--}$ Set $a_1 = s_2 q_1 + c_2 z_1$
# | 48. | $\phantom{--}$ Set $b_2 = s_2 z_2$
# | 49. | $\phantom{--}$ For $j = 2, 3, \ldots, n-1$ do Steps 50-51
# | 50. | $\phantom{----}$ $a_j = s_{j+1}q_j + c_j c_{j+1} z_j$
# | 51. | $\phantom{----}$ $b_{j+1} = s_{j+1} z_{j+1}$
# | 52. | $\phantom{--}$ Set $a_n = c_n z_n$
# | 53. | $\phantom{--}$ Set $k = k+1$
# | 54. | $\phantom{--}$ If $k = M$ then do Step 55
# | 55. | $\phantom{----}$ Print "Number of iteration exceeded"

# ### Python/NumPy implementation of QR eigenvalue algorithm
# 
# The pseudocode above exploits the tridiagonal structure of $\mathbf{A}$ to perform the $\mathbf{QR}$ factorization row-by-row in an efficient manner without using matrix multiplication operations.  However, a likely user of Python will simply use the built in `numpy.linalg` packaged functions to compute eigenvalues (which use the $\mathbf{QR}$ algorithm); therefore, the implementation below simply uses the matrix operations for clarity in the method.

# In[10]:


def QR(A):

    Q = gram_schmidt(A)
    R = Q.T @ A

    return (Q, R)


def eigs_2x2(A):

    b = -(A[-1, -1] + A[-2, -2])
    c = A[-1, -1] * A[-2, -2] - A[-2, -1] * A[-1, -2]
    d = np.sqrt(b ** 2 - 4 * c)

    if b > 0:
        return (-2 * c / (b + d), -(b + d) / 2)
    else:
        return ((d - b) / 2, 2 * c / (d - b))


def QR_eigenvalues(A, max_iterations=10000, tolerence=1e-6):

    n = A.shape[0]
    ?? = np.zeros(n, dtype=np.double)

    for _ in range(max_iterations):

        # Check to see if we have converged, then deflate by 1
        if np.abs(A[-1, -2]) <= tolerence:
            n -= 1
            ??[n] = A[-1, -1]
            A = A[:-1, :-1]

        # Here we are finding the eigenvalues of the lower 2 x 2 submatrix to
        # find the best value to shift by to improve convergence rates.
        ??1, ??2 = eigs_2x2(A)

        # Since we have a explicit formula for 2x2 case, let's use it
        if n == 2:
            ??[0] = ??1
            ??[1] = ??2
            break

        p = np.array([??1 - A[-1, -1], ??2 - A[-1, -1]]).argmin()

        ?? = ??1 if p == 0 else ??2

        I = np.eye(n)
        # Shifted QR decomp
        Q, R = QR(A - ?? * I)
        # Reverse QR
        A = R @ Q + ?? * I

    return ??


# In[11]:


get_ipython().run_cell_magic('javascript', '', 'function hideElements(elements, start) {\nfor(var i = 0, length = elements.length; i < length;i++) {\n    if(i >= start) {\n        elements[i].style.display = "none";\n    }\n}\n}\nvar prompt_elements = document.getElementsByClassName("prompt");\nhideElements(prompt_elements, 0)')

