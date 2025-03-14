import numpy as np


class SVM:
    def __init__(
        self, C=1.0, kernel="linear", lr=0.01, tol=1e-6, max_iter=100, mode="primal"
    ):
        """
        Initialize the SVM model.
        """

        self.C = C  # Regularization parameter
        self.kernel = kernel
        self.lr = lr  # Learning rate
        self.tol = tol  # Tolerance for stopping criterion
        self.max_iter = max_iter  # Max iterations
        self.mode = mode
        self.w = None  # Weight vector
        self.b = 0  # Bias term
        self.alpha = None  # Lagrange multipliers (for dual)
        self.support_vectors = None  # Support vectors (for dual)
        self.support_y = None  # Support vector labels (for dual)

    def fit(self, X, y, tau=0.001):
        """
        Train the SVM model.

        Args:
            X (numpy array): Feature matrix (M x d).
            y (numpy array): Labels (-1 or +1).
        """

        if self.mode == "primal":
            self._fit_primal(X, y)
        elif self.mode == "dual":
            self._fit_dual(X, y, tau)  # Placeholder for future dual implementation
        else:
            raise ValueError("Mode must be 'primal' or 'dual'.")

    def _fit_primal(self, X, y):
        """Train the SVM using Stochastic Gradient Descent (SGD) for the primal formulation."""

        M, d = X.shape  # Number of samples, number of features
        self.w = np.zeros(d)  # Initialize weights
        self.b = 0  # Initialize bias term

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

    def _fit_dual(self, X, y, d_intial: float, tau: float = 0.001):
        """
        Train the SVM using Projected Gradient Descent (PGD) for the dual formulation.
        """
        M = X.shape[0]
        G = self.compute_gram_matrix(X)  # Compute the Gram matrix
        Y = np.diag(y)
        self.alpha = np.zeros(M)  # Initialize Lagrange multipliers

        iter_count = 0
        diff = float("inf")

        while diff > self.tol and iter_count < self.max_iter:
            # Compute the gradient
            gradient = Y @ G @ Y @ self.alpha - np.ones_like(self.alpha)

            # Update the Lagrange multipliers
            alpha_new = self.project_onto_feasible_set(
                self.alpha - tau * gradient, y, self.C
            )

            # Step length selection (Barzilai–Borwein method)
            tau = self.step_length_selection(G, Y, self.alpha, alpha_new)
            print(f"Iteration {iter_count}: Step length = {tau}")

            # Check for convergence
            diff = np.linalg.norm(alpha_new - self.alpha)
            print(f"Iteration {iter_count}: Difference between alpha updates = {diff}")

            self.alpha = alpha_new
            iter_count += 1

        # Compute the weight vector
        self.w = np.sum((self.alpha[:, None] * y[:, None]) * X, axis=0)

        # Compute the bias term
        support_idx = (self.alpha > 1e-5) & (self.alpha < self.C)
        self.b = np.mean(y[support_idx] - np.dot(X[support_idx], self.w))

        self.support_vectors = X[support_idx]

        self.support_y = y[support_idx]

        return self.w, self.b

    def step_length_selection(self, G, Y, alpha: float, alpha_new: float) -> float:
        """
        Select the step length for the projected gradient descent.
        """

        tau_min = 1e-3
        tau_max = 10

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

    def alpha_lambda(self, beta, y, lambd, C):
        """
        Compute alpha(lambda) based on the given formula.
        """
        return np.minimum(np.maximum(beta + lambd * y, 0), C)

    def inner_product(self, y, alpha_lambda):
        """
        Compute the inner product <y, alpha(lambda)>.
        """
        return np.dot(y, alpha_lambda)

    def project_onto_feasible_set(self, gamma, y, C, delta=0.01):
        """
        Project the alpha vector onto the feasible set using bisection.
        """
        beta = np.copy(gamma)
        lambda_min, lambda_max = -10, 10

        while self.inner_product(y, self.alpha_lambda(beta, y, lambda_min, C)) > 0:
            lambda_min -= delta
            # lambda_max = lambda_min

        while self.inner_product(y, self.alpha_lambda(beta, y, lambda_max, C)) < 0:
            lambda_max += delta
            # lambda_min = lambda_max

        for _ in range(self.max_iter):
            lambda_hat = 0.5 * (lambda_min + lambda_max)
            inner_prod = self.inner_product(
                y, self.alpha_lambda(beta, y, lambda_hat, C)
            )

            if abs(inner_prod) < self.tol:
                return self.alpha_lambda(beta, y, lambda_hat, C)
            elif inner_prod < 0:
                lambda_min = lambda_hat
            else:
                lambda_max = lambda_hat

        return self.alpha_lambda(beta, y, lambda_hat, C)

    def predict(self, X):
        """
        Predict class labels using the trained SVM.
        """

        return np.sign(self.decision_function(X))

    def decision_function(self, X):
        """
        Compute decision function values for given data points in both Primal and Dual SVM.

        Parameters:
        - X: Test feature matrix (M_test x d)

        Returns:
        - Decision values: f(X) (M_test x 1)
        """
        if self.mode == "primal":
            return np.dot(X, self.w) + self.b

        elif self.mode == "dual":
            support_idx = (self.alpha > 1e-5) & (
                self.alpha < self.C
            )  # Mask for support vectors
            support_alphas = self.alpha[support_idx]
            support_vectors = self.support_vectors  # Already filtered in `_fit_dual`
            support_y = self.support_y  # Already filtered in `_fit_dual`

            decision_values = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                decision_values[i] = (
                    np.sum(
                        support_alphas
                        * support_y
                        * np.array(
                            [self._kernel_function(sv, X[i]) for sv in support_vectors]
                        )
                    )
                    + self.b
                )

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
