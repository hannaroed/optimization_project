import numpy as np


class SVM:
    def __init__(
        self, C=1.0, kernel="linear", lr=0.01, tol=1e-6, max_iter=1000, mode="primal"
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
        self.support_alphas = None  # Support vector Lagrange multipliers (for dual)

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
        M_1 = np.ones(M)
        self.alpha = np.zeros(M)  # Initialize Lagrange multipliers
        f_ref = float("inf")  # Reference function value
        f_best = self.inner_product(
            self.alpha, Y @ G @ Y @ self.alpha
        ) - self.inner_product(
            M_1, self.alpha
        )  # Best function value
        f_c = f_best  # Current function value
        L = 10
        l = 0
        theta_k = 1

        iter_count = 0
        diff = float("inf")

        while diff > self.tol and iter_count < self.max_iter:
            # Compute the gradient
            gradient = Y @ G @ Y @ self.alpha - np.ones_like(self.alpha)

            # Update the Lagrange multipliers
            alpha_new = self.project_onto_feasible_set(
                self.alpha - tau * gradient, y, self.C
            )
            d_k = alpha_new - self.alpha

            theta_star = self.inner_product(M_1, d_k) - self.inner_product(
                d_k, Y @ G @ Y @ alpha_new
            ) / (self.inner_product(d_k, Y @ G @ Y @ d_k) + 1e-8)
            print(f"Iteration {iter_count}: theta_star = {theta_star}")

            f_k = self.inner_product(
                alpha_new, Y @ G @ Y @ alpha_new
            ) - self.inner_product(M_1, alpha_new)

            theta_k = max(0, min(1, theta_star))

            if f_k < f_best:
                f_best = f_k
                f_c = f_k
                l = 0

            if f_k >= f_best:
                f_c = max(f_c, f_k)
                l += 1

            if l == L:
                f_ref = f_c
                f_c = f_k
                l = 0

            if (
                self.inner_product(alpha_new + d_k, Y @ G @ Y @ alpha_new)
                - self.inner_product(M_1, alpha_new)
                > f_ref
            ):
                print("Line search triggered")
                alpha_new = self.alpha + theta_k * d_k

            # Step length selection (Barzilai–Borwein method)
            tau = self.step_length_selection(G, Y, self.alpha, alpha_new)
            print(f"Iteration {iter_count}: Step length = {tau}")

            # Check for convergence
            diff = np.linalg.norm(
                self.project_onto_feasible_set(alpha_new - gradient, y, self.C)
                - alpha_new
            )
            print(f"Iteration {iter_count}: Difference between alpha updates = {diff}")

            self.alpha = alpha_new
            iter_count += 1

        support_idx = (self.alpha > 1e-5) & (self.alpha < self.C)

        self.support_vectors = X[support_idx]

        self.support_y = y[support_idx]

        self.support_alphas = self.alpha[support_idx]

        # Compute the weight vector
        self.w = np.sum(
            (self.support_alphas[:, None] * self.support_y[:, None])
            * self.support_vectors,
            axis=0,
        )

        # Compute the bias term
        self.b = np.mean(self.support_y - np.dot(self.support_vectors, self.w))

        print(
            f"Converged after {iter_count} iterations: Difference between alpha updates = {diff}"
        )

        return self.w, self.b

    def step_length_selection(self, G, Y, alpha: float, alpha_new: float) -> float:
        """
        Select the step length for the projected gradient descent using Barzilai-Borwein
        """

        tau_min = 1e-3
        tau_max = 0.1

        # Compute gradient at alpha^(k)
        grad_alpha = Y @ G @ Y @ alpha - np.ones_like(alpha)

        # Compute gradient at alpha^(k+1)
        grad_alpha_new = Y @ G @ Y @ alpha_new - np.ones_like(alpha_new)

        # Compute differences s and z
        s = alpha_new - alpha
        z = grad_alpha_new - grad_alpha

        denom = np.dot(s, z) + 1e-8

        if np.dot(s, z) <= 0:
            return tau_max
        else:
            tau = np.dot(s, s) / denom
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

    def bracketing_phase(
        self, alpha, y, C, delta=0.01, lambda_val=0, lambda_min=None, lambda_max=None
    ):
        """
        Perform the bracketing phase for bisection.
        """
        alpha_projected = self.alpha_lambda(alpha, y, 0, C)

        r = self.inner_product(y, alpha_projected)

        # Bracketing phase
        if r < 0:
            lambda_min = lambda_val
            r_min = r
            lambda_val += delta
            alpha_lambda = self.alpha_lambda(alpha, y, lambda_val, C)
            r = self.inner_product(y, alpha_lambda)
            while r < 0:
                lambda_min = lambda_val
                r_min = r
                s = max(r_min / r - 1, 0.1)
                delta += delta / s
                lambda_val += delta
                alpha_lambda = self.alpha_lambda(alpha, y, lambda_val, C)
                r = self.inner_product(y, alpha_lambda)

            lambda_max = lambda_val
            r_max = r
        else:
            lambda_max = lambda_val
            r_max = r
            lambda_val -= delta
            alpha_lambda = self.alpha_lambda(alpha, y, lambda_val, C)
            r = self.inner_product(y, alpha_lambda)
            while r > 0:
                lambda_max = lambda_val
                r_max = r
                s = max(r_max / r - 1, 0.1)
                delta += delta / s
                lambda_val -= delta
                alpha_lambda = self.alpha_lambda(alpha, y, lambda_val, C)
                r = self.inner_product(y, alpha_lambda)

            lambda_min = lambda_val
            r_min = r
        return lambda_val, lambda_min, lambda_max, r_min, r_max

    def secant_phase(
        self, alpha, y, C, delta, r_min, r_max, lambda_val, lambda_min, lambda_max
    ):
        """
        Perform the secant phase for bisection.
        """
        s = 1 - r_min / r_max
        delta = delta / s
        lambda_val = lambda_max - delta
        r = self.inner_product(y, self.alpha_lambda(alpha, y, lambda_val, C))
        while abs(r) > self.tol:
            if r > 0:
                if s <= 2:
                    lambda_max = lambda_val
                    r_max = r
                    s = 1 - r_min / r_max
                    delta = (lambda_max - lambda_min) / s
                    lambda_val = lambda_max - delta
                else:
                    s = max(r_max / r - 1, 0.1)
                    delta = (lambda_max - lambda_val) / s
                    lambda_new = max(
                        lambda_val - delta, 0.75 * lambda_min + 0.25 * lambda_val
                    )
                    lambda_max = lambda_val
                    r_max = r
                    lambda_val = lambda_new
                    s = (lambda_max - lambda_min) / (lambda_max - lambda_val)
            else:
                if s >= 2:
                    lambda_min = lambda_val
                    r_min = r
                    s = 1 - r_min / r_max
                    delta = (lambda_max - lambda_min) / s
                    lambda_val = lambda_max - delta
                else:
                    s = max(r_min / r - 1, 0.1)
                    delta = (lambda_val - lambda_min) / s
                    lambda_new = min(
                        lambda_val + delta, 0.75 * lambda_max + 0.25 * lambda_val
                    )
                    lambda_min = lambda_val
                    r_min = r
                    lambda_val = lambda_new
                    s = (lambda_max - lambda_min) / (lambda_max - lambda_val)

            r = self.inner_product(y, self.alpha_lambda(alpha, y, lambda_val, C))

        return lambda_val

    def project_onto_feasible_set(self, alpha, y, C, delta=0.01):
        """
        Project the alpha vector onto the feasible set using bisection.
        """
        # Bracketing phase
        lambda_val, lambda_min, lambda_max, r_min, r_max = self.bracketing_phase(
            alpha, y, C, delta
        )

        # Secant phase for more precise lambda selection
        lambda_val = self.secant_phase(
            alpha, y, C, delta, r_min, r_max, lambda_val, lambda_min, lambda_max
        )
        return self.alpha_lambda(alpha, y, lambda_val, C)

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
            decision_values = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                decision_values[i] = (
                    np.sum(
                        self.support_alphas
                        * self.support_y
                        * np.array(
                            [
                                self._kernel_function(sv, X[i])
                                for sv in self.support_vectors
                            ]
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
