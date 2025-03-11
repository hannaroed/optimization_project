import numpy as np

class SVM:
    def __init__(self, C=1.0, kernel="linear", alpha=0.01, tol=1e-4, max_iter=1000, mode="primal"):
        """
        Initialize the SVM model.
        """
        self.C = C
        self.kernel = kernel
        self.alpha = alpha
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
        M, d = X.shape # number of samples, number of features per sample
        self.w = np.zeros(d) # initialize weights
        self.b = 0 # initialize bias term
        for _ in range(self.max_iter):
            for i in range(M):
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)
                if margin < 1:
                    self.w = self.w - self.lr * self.w + self.lr * (self.C * y[i] * X[i])  # weight update
                    self.b = self.b + self.lr * (self.C * y[i]) # bias update
                else:
                    self.w = self.w - self.alpha * self.w # apply weight decay
        return self.w, self.b

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