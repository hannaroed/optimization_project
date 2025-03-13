import numpy as np

class SVM:
    def __init__(self, C=1.0, kernel="linear", lr=0.01, tol=1e-4, max_iter=1000, mode="primal"):
        """
        Initialize the SVM model.
        """
        self.C = C # Regularization parameter
        self.kernel = kernel
        self.lr = lr # Learning rate
        self.tol = tol # Tolerance for stopping criterion
        self.max_iter = max_iter # Max iterations
        self.mode = mode
        self.w = None # Weight vector (for primal)
        self.b = 0 # Bias term
        self.alpha = None # Lagrange multipliers (for dual)
        self.support_vectors = None # Support vectors (for dual)
        self.support_y = None # Support vector labels (for dual)

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
        M, d = X.shape # Number of samples, number of features
        self.w = np.zeros(d) # Initialize weights
        self.b = 0 # Initialize bias term

        for _ in range(self.max_iter):
            for i in range(M):
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)

                if margin < 1:
                    # Update for misclassified points (hinge loss gradient)
                    self.w = (1 - self.lr) * self.w + self.lr * self.C * y[i] * X[i]
                    self.b += self.lr * self.C * y[i]
                else:
                    # Update for correctly classified points (regularization)
                    self.w = (1 - self.lr) * self.w
        
        return self.w, self.b

    def _fit_dual(self, X, y):
        """
        Train the SVM using Projected Gradient Descent (PGD) for the dual formulation.
        """
        pass

    def compute_gram_matrix(self, X):
        """
        Compute the Gram matrix for the given data points.
        """
        M = X.shape[0]
        G = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                G[i, j] = self._kernel_function(X[i], X[j])
        return G

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
            pass
        else:
            raise ValueError("Mode must be 'primal' or 'dual'.")

    def _kernel_function(self, x1, x2):
        """
        Compute the kernel function.
        """
        if self.kernel == "linear":
            return np.dot(x1, x2)
        else:
            raise ValueError("Only 'linear' kernel is supported so far.")