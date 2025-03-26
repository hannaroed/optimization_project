import numpy as np
from scipy.optimize import minimize_scalar

class SVM:
    def __init__(self, C=1.0, kernel="linear", lr=0.01, tol=1e-6, max_iter=1000, mode="primal_SGD", sigma=1.0, s=1.0):
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
        self.support_alphas = None # Support vector Lagrange multipliers (for dual)
        self.sigma = sigma # Kernel bandwidth
        self.s = s # Kernel parameter

        self.tau_min = 1e-10
        self.tau_max = 1e10
        self._s_prev = None
        self._z_prev = None

    def fit(self, X, y, tau=0.001):
        """
        Train the SVM model.

        Args:
            X (numpy array): Feature matrix (M x d).
            y (numpy array): Labels (-1 or +1).
        """

        if self.mode == "primal_SGD":
            self._fit_primal_SGD(X, y)
        elif self.mode == "primal_QP":
            self._fit_primal_QP(X, y)
        elif self.mode == "dual":
            self._fit_dual(X, y)
        else:
            raise ValueError("Mode must be 'primal_SGD', 'primal_QP' or 'dual'.")

    def _fit_primal_SGD(self, X, y):
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

    def _fit_primal_QP(self, X, y):
        """
        Train the SVM using the Quadratic Penalty (QP) method for the primal formulation."""

        M, d = X.shape
        w = np.zeros(d)
        b = 0.0
        mu_k = 1.0  # Initial penalty parameter
        tau_k = self.tol  # Use tolerance from the class

        for k in range(self.max_iter):
            # Compute the penalized objective function Q(w, b, mu_k)
            margin = y * (np.dot(X, w) + b)
            hinge_loss = np.maximum(0, 1 - margin) ** 2  # Quadratic penalty term

            # Compute gradients
            indicator = margin < 1
            grad_w = w - self.C * np.sum((2 * (1 - margin) * y * X.T * indicator), axis=1)
            grad_b = -self.C * np.sum(2 * (1 - margin) * y * indicator)

            # Update weights and bias using gradient descent
            w -= self.lr * grad_w
            b -= self.lr * grad_b

            # Check stopping condition
            grad_norm = np.linalg.norm(np.append(grad_w, grad_b))
            if grad_norm <= tau_k:
                break  # Convergence reached

            # Increase penalty parameter and tighten tolerance
            mu_k *= 10
            tau_k *= 0.1

        self.w = w
        self.b = b

        return self.w, self.b

    def _fit_dual(self, X, y):
        """
        Train the SVM using Projected Gradient Descent (PGD) with adaptive step length (Barzilai-Borwein) for the dual formulation.
        """
        M = X.shape[0]
        G = self._compute_gram_matrix(X)  # Compute the Gram matrix
        self.alpha = np.zeros(M)  # Initialize Lagrange multipliers

        iter_count = 0
        diff = float("inf")
        tau = 1.0  # Initial step size (add line search)

        # Reset step history
        self._s_prev = None
        self._z_prev = None

        while diff > self.tol and iter_count < self.max_iter:
            # Compute gradient of the dual objective
            gradient = y * (G @ (y * self.alpha)) - 1

            # Projected gradient step
            alpha_new = self._project_onto_feasible_set(
                self.alpha - tau * gradient, y, self.C
            )

            # Compute difference for convergence check
            diff = np.linalg.norm(alpha_new - self.alpha)

            # Update step length using enhanced BB method
            tau = self._step_length_selection(G, y, self.alpha, alpha_new)

            # Update alpha and iteration counter
            self.alpha = alpha_new
            iter_count += 1

            print(f"Iteration {iter_count}: τ = {tau:.4e}, Δα = {diff:.4e}")

        # Identify support vectors
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

        print(f"Converged after {iter_count} iterations: Δα = {diff:.4e}")

        return self.w, self.b

    def _step_length_selection(self, G, y, alpha_old, alpha_new):
        """
        Enhanced Barzilai-Borwein step size using current and previous s, z values.
        """
        s = alpha_new - alpha_old
        z = y * (G @ (y * s))  # Gradient difference approximation

        # Store current s and z for next iteration
        if self._s_prev is not None and self._z_prev is not None:
            num = np.dot(s, s) + np.dot(self._s_prev, self._s_prev)
            denom = np.dot(s, z) + np.dot(self._z_prev, self._z_prev)
        else:
            num = np.dot(s, s)
            denom = np.dot(s, z)

        self._s_prev = s
        self._z_prev = z

        if denom <= 1e-10:
            return self.tau_max

        tau = num / denom
        print(f"τ = {tau:.4e}")
        tau = min(max(tau, self.tau_min), self.tau_max)

        return tau

    def _compute_gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the Gram matrix for the given data points.
        """

        # X: (M x d)

        G = self._kernel_function(X[:, None, ...], X[None, :, ...])
        # G: (M x M)
        return G

    def _alpha_lambda(self, beta, y, lambd, C):
        """
        Compute alpha(lambda) based on the given formula.
        """
        return np.minimum(np.maximum(beta + lambd * y, 0), C)

    def _inner_product(self, y, alpha_lambda):
        """
        Compute the inner product <y, alpha(lambda)>.
        """
        return np.sum(y * alpha_lambda, axis=-1)

    def _bracketing_phase(
        self, alpha, y, C, delta=0.01, lambda_val=0, lambda_min=None, lambda_max=None
    ):
        """
        Perform the bracketing phase for bisection.
        """
        alpha_projected = self._alpha_lambda(alpha, y, 0, C)

        r = self._inner_product(y, alpha_projected)

        # Bracketing phase
        if r < 0:
            lambda_min = lambda_val
            r_min = r
            lambda_val += delta
            alpha_lambda = self._alpha_lambda(alpha, y, lambda_val, C)
            r = self._inner_product(y, alpha_lambda)
            while r < 0:
                lambda_min = lambda_val
                r_min = r
                s = max(r_min / r - 1, 0.1)
                delta += delta / s
                lambda_val += delta
                alpha_lambda = self._alpha_lambda(alpha, y, lambda_val, C)
                r = self._inner_product(y, alpha_lambda)

            lambda_max = lambda_val
            r_max = r
        else:
            lambda_max = lambda_val
            r_max = r
            lambda_val -= delta
            alpha_lambda = self._alpha_lambda(alpha, y, lambda_val, C)
            r = self._inner_product(y, alpha_lambda)
            while r > 0:
                lambda_max = lambda_val
                r_max = r
                s = max(r_max / r - 1, 0.1)
                delta += delta / s
                lambda_val -= delta
                alpha_lambda = self._alpha_lambda(alpha, y, lambda_val, C)
                r = self._inner_product(y, alpha_lambda)

            lambda_min = lambda_val
            r_min = r
        return lambda_val, lambda_min, lambda_max, r_min, r_max

    def _secant_phase(
        self, alpha, y, C, delta, r_min, r_max, lambda_val, lambda_min, lambda_max
    ):
        """
        Perform the secant phase for bisection.
        """
        s = 1 - r_min / r_max
        delta = delta / s
        lambda_val = lambda_max - delta
        r = self._inner_product(y, self._alpha_lambda(alpha, y, lambda_val, C))
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

            r = self._inner_product(y, self._alpha_lambda(alpha, y, lambda_val, C))

        return lambda_val

    def _project_onto_feasible_set(self, alpha, y, C, delta=0.01):
        """
        Project the alpha vector onto the feasible set using bisection.
        """
        # Bracketing phase
        lambda_val, lambda_min, lambda_max, r_min, r_max = self._bracketing_phase(
            alpha, y, C, delta
        )

        # Secant phase for more precise lambda selection
        lambda_val = self._secant_phase(
            alpha, y, C, delta, r_min, r_max, lambda_val, lambda_min, lambda_max
        )

        return self._alpha_lambda(alpha, y, lambda_val, C)

    def predict(self, X):
        """
        Predict class labels using the trained SVM.
        """

        return np.sign(self._decision_function(X))

    def _decision_function(self, X):
        """
        Compute decision function values for given data points in both Primal and Dual SVM.

        Parameters:
        - X: Test feature matrix (M_test x d)

        Returns:
        - Decision values: f(X) (M_test x 1)
        """
        if self.mode == "primal_SGD" or self.mode == "primal_QP":
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
            raise ValueError("Mode must be 'primal_SGD', 'primal_QP' or 'dual'.")

    def _kernel_function(self, x1, x2):
        """
        Compute the kernel function.
        """

        if self.kernel == "linear":
            return np.sum(x1 * x2, axis=-1)
        
        elif self.kernel == "gaussian":
            return np.exp(-np.linalg.norm(x1 - x2, axis=-1) ** 2 / (2 * self.sigma ** 2))

        elif self.kernel == "laplacian":
            return np.exp(-np.linalg.norm(x1 - x2, axis=-1) / self.sigma)
        
        elif self.kernel == "inverse multiquadratic":
            return 1 / (self.sigma ** 2 + np.linalg.norm(x1 - x2, axis=-1) ** 2)**self.s

        else:
            raise ValueError("Unsupported kernel function.")

if __name__ == '__main__':
    import random
    from test_data import TestLinear
    w = np.array([1.0, 1.0])
    b = 1.0
    n_A = 300
    n_B = 200
    margin = 0.5

    random_seed = random.randint(0, 1000)
    print(f"Using random seed: {random_seed}")

    listA, listB = TestLinear(w, b, n_A, n_B, margin, seed=random_seed)

    # Convert lists to numpy arrays
    X_A = np.array(listA)
    X_B = np.array(listB)
    X = np.vstack((X_A, X_B))
    y = np.hstack((np.ones(n_A), -np.ones(n_B)))  # Class A = +1, Class B = -1

    # Train the SVM
    svm = SVM(C=1.0, kernel="linear", lr=0.01, mode="dual", sigma=1.5, s=1.0)
    svm.fit(X, y)

    # Predict decision boundary
    xx, yy = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
    Z = np.c_[xx.ravel(), yy.ravel()]
    preds = svm.predict(Z).reshape(xx.shape)
    print("Finished predict")