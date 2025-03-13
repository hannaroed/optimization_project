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
        self.w = None # Weight vector
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

    def _fit_dual(self, X, y, d_intial: float, tau: float):
        """
        Train the SVM using Projected Gradient Descent (PGD) for the dual formulation.
        """
        M = X.shape[0]
        G = self.compute_gram_matrix(X) # Compute the Gram matrix
        Y = np.diag(y)
        self.alpha = np.zeros(M) # Initialize Lagrange multipliers

        d = d_intial
        iter_count = 0
        diff = float('inf')

        while diff > self.tol and iter_count < self.max_iter:

            # Compute the gradient
            gradient = Y @ G @ Y @ self.alpha - np.ones_like(self.alpha)

            # Update the Lagrange multipliers
            alpha_new = self.project_onto_feasible_set(self.alpha - tau * gradient, y, self.C)

            # Step length selection (Barzilai–Borwein method)
            tau = self.step_length_selection(G, Y, self.alpha, alpha_new)

            # Check for convergence
            diff = np.linalg.norm(alpha_new - self.alpha)
            self.alpha = alpha_new
            iter_count += 1

        # Compute the weight vector
        self.w = np.sum(self.alpha * y[:, None] * X, axis=0)

        # Compute the bias term
        support_idx = (self.alpha > 1e-5) & (self.alpha < self.C)
        self.b = np.mean(y[support_idx] - np.dot(X[support_idx], self.w))

        return self.w, self.b
    
    def step_length_selection(self, G, Y, alpha: float, alpha_new: float) -> float:
        """
        Select the step length for the projected gradient descent.
        """

        tau_min = 10e-5
        tau_max = 10e5

        # Compute gradient at alpha^(k)
        grad_alpha = Y @ G @ Y @ alpha - np.ones_like(alpha)

        # Compute gradient at alpha^(k+1)
        grad_alpha_new = Y @ G @ Y @ alpha_new - np.ones_like(alpha_new)

        # Compute differences s and z
        s = alpha_new - alpha 
        z = grad_alpha_new - grad_alpha
        
        if np.dot(s, z) <= 0: 
            return tau_max
        else:
            tau = np.dot(s, s) / np.dot(s, z)
            tau = max(tau_min, min(tau, tau_max))
        return tau

    def compute_gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the Gram matrix for the given data points.
        """

        M = X.shape[0]
        G = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                G[i, j] = self._kernel_function(X[i], X[j])
        return G
    
    def project_onto_feasible_set(self, alpha, y, C, tol=1e-6, delta=0.1):
        """
        Project the alpha vector onto the feasible set.
        """
    
        beta = np.copy(alpha)
        lambda_min, lambda_max = -10, 10

        def alpha_lambda(lambda_val):
            return np.clip(beta + lambda_val * y, 0, C)
        
        if np.dot(y, alpha_lambda(lambda_min)) > 0:
            while np.dot(y, alpha_lambda(lambda_min)) >= 0 & np.dot(y, alpha_lambda(lambda_max)) <= 0:
                lambda_min -= delta
                lambda_max = lambda_min
        if np.dot(y, alpha_lambda(lambda_max)) < 0:
            while np.dot(y, alpha_lambda(lambda_max)) <= 0 & np.dot(y, alpha_lambda(lambda_min)) >= 0:
                lambda_max += delta
                lambda_min = lambda_max

        lambda_mid = (lambda_max + lambda_min) / 2
        
        while np.abs(np.dot(y, alpha_lambda(lambda_mid))) >= tol:
    
            if np.dot(y, alpha_lambda(lambda_mid)) > 0:
                lambda_min = lambda_mid
            else:
                lambda_max = lambda_mid

            lambda_mid = (lambda_max + lambda_min) / 2

        projected_alpha = alpha_lambda(lambda_mid)

        return projected_alpha
     

    def predict(self, X):
        """
        Predict class labels using the trained SVM.
        """

        return np.sign(self.decision_function(X))

    def decision_function(self, X):
        """
        Compute decision function values for given data points in Dual SVM.

        Parameters:
        - X: Test feature matrix (M_test x d)

        Returns:
        - Decision values: f(X) (M_test x 1)
        """

        if self.mode == "primal":
            return np.dot(X, self.w) + self.b

        elif self.mode == "dual":
            if self.alpha is None or self.support_vectors is None:
                raise ValueError("Model is not trained yet. Call fit() first.")

            # Compute decision function
            K_x = np.array([[self._kernel_function(x_i, x_j) for x_j in X] for x_i in self.support_vectors])
            decision_values = np.dot((self.alpha * self.support_y), K_x) + self.b

            return decision_values

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