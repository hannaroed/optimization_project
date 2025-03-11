import numpy as np

class SVM:
    def __init__(self, C=1.0, kernel="linear", lr=0.01, tol=1e-4, max_iter=1000, mode="primal"):
        """
        Initialize the SVM model.
        """
        self.C = C
        self.kernel = kernel
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.mode = mode
        self.w = None  # Primal weights
        self.b = 0  # Bias term
        self.alpha = None  # Dual variables (to be used later)
        self.support_vectors = None  # Support vectors (to be used in dual)
        self.support_y = None  # Support vector labels (for dual)

    def fit(self, X, y):
        """
        Train the SVM model.

        Args:
            X (numpy array): Feature matrix (M x d).
            y (numpy array): Labels (-1 or +1).
        """
        if self.mode == "primal":
            self._fit_primal(X, y)
        elif self.mode == "dual":
            self._fit_dual(X, y)  # Placeholder for future dual implementation
        else:
            raise ValueError("Mode must be 'primal' or 'dual'.")

    def _fit_primal(self, X, y):
        """Train the SVM using Stochastic Gradient Descent (SGD) for the primal formulation."""
    pass

    def _fit_dual(self, X, y):
        """
        Train the SVM using Projected Gradient Descent (PGD) for the dual formulation.
        """
        pass

    def predict(self, X):
        """
        Predict class labels using the trained SVM.

        """
        return np.sign(self.decision_function(X))

    def decision_function(self, X):
        """
        Compute decision function values for given data points.

        """
        if self.mode == "primal":
            return np.dot(X, self.w) + self.b
        elif self.mode == "dual":
            raise NotImplementedError("Decision function for dual SVM not implemented yet.")
        else:
            raise ValueError("Mode must be 'primal' or 'dual'.")

    def _kernel_function(self, x1, x2):
        """
        Compute the kernel function.
        """
        if self.kernel == "linear":
            return np.dot(x1, x2)
        else:
            raise ValueError("Only 'linear' kernel is supported in primal SVM.")