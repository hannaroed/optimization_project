import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


def TestLinear(w, b, n_A, n_B, margin, **kwargs):
    """


    Parameters
    ----------
    w : non-zero vector
        normal vector defining a hyperplane
    b : real number
        offset of the hyperplane
    n_A : integer
        number of additional samples from class A
    n_B: integer
        number of additional samples from class B
    margin : positive real
        desired margin for the samples

    Optional Parameters
    -------------------
    seed : integer
        seed for the random number generator
        default value : 18
    sigma : positive real
        standard deviation for the normal distribution
        default value : 1.
    shape : positive real
        shape parameter for the Gamma distribution
        default value : 1.
    scale : positive real
        scale parameter for the Gamma distribution
        default value : 1.

    Returns
    -------
    list_A, list_B : lists of vectors
        list_A contains n_A vectors all lying on one side of the hyperplane H(w,-b).
        The distance to the hyperplane is margin + a sample of a Gamma distribution.
        In the plane normal to w, the points follow a normal distribution.
        One of the vectors acts as a support vector with precise margin gamma.
        list_B contains n_B vectors, produced in a similar way, lying on the
        opposite side of the hyperplane.

    """

    # read out additional keyword arguments
    seed = kwargs.get("seed", 18)
    shape = kwargs.get("shape", 1.0)
    scale = kwargs.get("scale", 1.0)
    sigma = kwargs.get("sigma", 1.0)

    # read out the number of dimensions
    d = w.size

    # rescale w to length 1
    norm_w = np.linalg.norm(w)
    w = w / norm_w
    b = b / norm_w

    # initialise a random number generator
    rng = default_rng(seed)

    # initialise an empty list
    list_A = []
    # draw samples for class A
    for _ in range(n_A):
        # draw n_A samples of a d-dimensional normal distribution
        vec = rng.normal(size=d, scale=sigma)
        # draw n_A samples of a Gamma distribution
        dist = rng.gamma(shape, scale)
        # project vec onto w^\perp
        vec += -np.inner(vec, w) * w
        # add (dist+margin+b)*w to vec
        vec += (dist + margin - b) * w
        # append the vector vec to list_A
        list_A.append(vec)

    # initialise an empty list
    list_B = []
    # draw samples for class A
    for _ in range(n_B):
        # draw n_B samples of a d-dimensional normal distribution
        vec = rng.normal(size=d, scale=sigma)
        # draw n_A samples of a Gamma distribution
        dist = rng.gamma(shape, scale)
        # project vec onto w^\perp
        vec += -np.inner(vec, w) * w
        # add -(dist+margin-b)*w to vec
        vec += (-b - dist - margin) * w
        # append the vector vec to list_B
        list_B.append(vec)

    # choose a random vector of each list and force it to be a support vector
    vec = rng.normal(size=d, scale=sigma)
    vec += -np.inner(vec, w) * w
    supp_A = rng.integers(0, n_A)
    list_A[supp_A] = vec + (margin - b) * w
    supp_B = rng.integers(0, n_B)
    list_B[supp_B] = vec + (-b - margin) * w

    return (list_A, list_B)


def TestNonLinear(
    n_A, n_B, margin=5.0, n_clusters=3, cluster_spread=0.8, plot_extent=8.0, **kwargs
):
    """
    Generate two classes as multiple Gaussian clusters (non-linear separability).

    Parameters
    ----------
    n_A : int
        Number of samples in class A.
    n_B : int
        Number of samples in class B.
    margin : float
        Minimum Euclidean distance between class A and B cluster centers.
    n_clusters : int
        Number of clusters per class.
    cluster_spread : float
        Standard deviation of each cluster.
    plot_extent : float
        Range of space within which cluster centers are sampled.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_A : (n_A, 2) np.ndarray
        Samples from class A.
    X_B : (n_B, 2) np.ndarray
        Samples from class B.
    """
    seed = kwargs.get("seed", 42)
    rng = default_rng(seed)

    samples_A = np.full(n_clusters, n_A // n_clusters)
    samples_B = np.full(n_clusters, n_B // n_clusters)

    # Handle any leftover points
    samples_A[: n_A % n_clusters] += 1
    samples_B[: n_B % n_clusters] += 1

    X_A = np.empty((0, 2))
    X_B = np.empty((0, 2))

    for n_a, n_b in zip(samples_A, samples_B):
        # Generate a center for class A
        center_A = rng.uniform(-plot_extent, plot_extent, size=2)
        cluster_A = rng.normal(loc=center_A, scale=cluster_spread, size=(n_a, 2))

        # Generate a center for class B that is at least 'margin' away from A
        while True:
            center_B = rng.uniform(-plot_extent, plot_extent, size=2)
            if np.linalg.norm(center_B - center_A) >= margin:
                break
        cluster_B = rng.normal(loc=center_B, scale=cluster_spread, size=(n_b, 2))

        # Stack clusters into final arrays
        X_A = np.vstack((X_A, cluster_A))
        X_B = np.vstack((X_B, cluster_B))

    return X_A, X_B


if __name__ == "__main__":
    w = np.array([1.0, 1.0])
    b = 1.0
    n_A = 10
    n_B = 8
    margin = 5.0e-1
    listA, listB = TestLinear(w, b, n_A, n_B, margin)
    [plt.scatter(x[0], x[1], color="r") for x in listA]
    [plt.scatter(x[0], x[1], color="b") for x in listB]
    plt.show()
