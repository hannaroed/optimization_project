import numpy as np
from scipy.optimize import minimize_scalar
from pprint import pprint
from typing import Tuple


class SVM:
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "linear",
        lr: float = 0.01,
        tol: float = 1e-6,
        max_iter: int = 1000,
        mode: str = "primal_SGD",
        sigma: float = 1.0,
        s: float = 1.0,
    ):
        """
        Initialize the SVM model.
        """

        self.C = C  # Regularization parameter
        self.kernel = kernel
        self.lr = lr  # Learning rate
        self.tol = tol  # Tolerance for stopping criterion
        self.max_iter = max_iter  # Max iterations
        self.mode = mode  # primal / dual
        self.w = None  # Weight vector
        self.b = 0  # Bias term
        self.alpha = None  # Lagrange multipliers (for dual)
        self.support_vectors = None  # Support vectors (for dual)
        self.support_y = None  # Support vector labels (for dual)
        self.support_alphas = None  # Support vector Lagrange multipliers (for dual)
        self.sigma = sigma  # Kernel bandwidth
        self.s = s  # Kernel parameter

        # Parameters for the Barzilai-Borwein method
        self.tau_min = 1e-5
        self.tau_max = 1e5
        self._s_prev = None
        self._z_prev = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Dispatcher function for training the SVM model.

        Args:
            X: Feature matrix (M x d).
            y: Labels (-1 or +1).

        Returns:
            None
        """

        if self.mode == "primal_SGD":
            self._fit_primal_SGD(X, y)
        elif self.mode == "primal_QP":
            self._fit_primal_QP(X, y)
        elif self.mode == "dual":
            self._fit_dual(X, y)
        else:
            raise ValueError("Mode must be 'primal_SGD', 'primal_QP' or 'dual'.")

    def _fit_primal_SGD(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Train the SVM using Stochastic Gradient Descent (SGD) for the primal formulation.

        Args:
            X: Feature matrix (M x d).
            y: Labels (-1 or +1).

        Returns:
            w, b: Weight and bias term vectors.
        """

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

    def _fit_primal_QP(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Train the SVM using the Quadratic Penalty (QP) method for the primal formulation.

        Args:
            X: Feature matrix (M x d).
            y: Labels (-1 or +1).

        Returns:
            w, b: Weight and bias term vectors.
        """

        _, d = X.shape
        w = np.zeros(d)
        b = 0.0
        mu_k = 1.0  # Initial penalty parameter
        tau_k = self.tol  # Use tolerance from the class

        for k in range(self.max_iter):
            # Compute the penalized objective function Q(w, b, mu_k)
            margin = y * (np.dot(X, w) + b)

            # Compute gradients
            indicator = margin < 1
            grad_w = w - self.C * np.sum(
                (2 * (1 - margin) * y * X.T * indicator), axis=1
            )
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

    def _fit_dual(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Train the SVM using Projected Gradient Descent (PGD),
        with adaptive step length (advanced Barzilai-Borwein combined with exact line search) for the dual formulation.

        Args:
            X: Feature matrix (M x d).
            y: Labels (-1 or +1).

        Returns:
            w, b: Weight and bias term vectors (if kernel != linear, only bias term has a value).
        """
        M = X.shape[0]
        G = self._compute_gram_matrix(X)  # Compute the Gram matrix
        self.alpha = np.zeros(M)  # Initialize Lagrange multipliers

        iter_count = 0
        diff = float("inf")  # parameter to measure convergence
        tau = 1.0  # Initial step size

        # Reset step history
        self._s_prev = None
        self._z_prev = None

        f_prev = self._dual_objective(G, y, self.alpha)
        f_best = f_prev

        f_c = f_best
        f_ref = float("inf")  # Start as infinity
        L = 10  # Number of allowed iterations without improvement
        non_improving_steps_counter = 0  # Counter for non-improving steps

        while diff > self.tol and iter_count < self.max_iter:
            # Compute gradient of the dual objective
            gradient = y * (G @ (y * self.alpha)) - 1

            # Projected gradient step
            alpha_after = self._project_onto_feasible_set(
                self.alpha - tau * gradient, y, self.C
            )

            f_before_step = f_prev
            f_after_step = self._dual_objective(G, y, alpha_after)

            if f_before_step < f_best:  # Improved using dual
                f_best = f_before_step
                f_c = f_before_step
                non_improving_steps_counter = 0
            else:  # Didn't improve
                f_c = max(f_c, f_before_step)
                non_improving_steps_counter += 1

            if non_improving_steps_counter == L:  # We have not improved for L steps
                f_ref = f_c
                f_c = f_before_step
                non_improving_steps_counter = 0

            if (
                f_after_step > f_ref or iter_count == 0
            ):  # Result is worse than the reference
                print(f"Iteration: {iter_count}, Line search triggered")
                # Perform exact line search in direction d = alpha_new - alpha
                d = alpha_after - self.alpha
                theta = self._exact_line_search(self.alpha, d, G, y)
                alpha_after = self._project_onto_feasible_set(
                    self.alpha + theta * d, y, self.C
                )
                f_after_step = self._dual_objective(
                    G, y, alpha_after
                )  # Recompute after update

            # Compute difference for convergence check
            diff = np.linalg.norm(alpha_after - self.alpha)

            # Update step length using enhanced BB method
            tau = self._step_length_selection(G, y, self.alpha, alpha_after)

            # Update alpha and iteration counter
            self.alpha = alpha_after
            f_prev = f_after_step
            iter_count += 1

        # Identify support vectors
        support_idx = (self.alpha > 1e-5) & (self.alpha < self.C)
        self.support_vectors = X[support_idx]
        self.support_y = y[support_idx]
        self.support_alphas = self.alpha[support_idx]

        if self.kernel == "linear":
            # Compute the weight vector
            self.w = np.sum(
                (self.support_alphas[:, None] * self.support_y[:, None])
                * self.support_vectors,
                axis=0,
            )

            # Bias term
            self.b = np.mean(self.support_y - np.dot(self.support_vectors, self.w))

        else:  # Non-linear kernel
            self.w = None  # No fixed weight vector

            # Bias term
            K_vals = self._kernel_function(
                self.support_vectors, self.support_vectors[0]
            )
            self.b = self.support_y[0] - np.sum(
                self.support_alphas * self.support_y * K_vals
            )

        return self.w, self.b

    def _exact_line_search(
        self, alpha: np.ndarray, d: float, G: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Perform exact line search in direction d by minimizing f(alpha + theta * d_k), wrt theta.

        Args:
            alpha: Lagrange parameter
            d: line search direction
            G: Gram matrix (M x M)
            y: Labels (-1 or +1).

        Returns:
            Minimized theta (res.x).
        """
        yd = y * d
        ya = y * alpha

        def f_theta(theta: float) -> float:
            return 0.5 * (ya + theta * yd) @ G @ (ya + theta * yd) - np.sum(
                alpha + theta * d
            )

        res = minimize_scalar(f_theta, bounds=(0, 1), method="bounded")  # minimizing
        return res.x

    def _dual_objective(self, G: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> float:
        """
        Compute dual objective value (f(alpha) = 0.5 * inner product(alpha, yGy * alpha) - inner product(1_M, alpha)).

        Args:
            G: Gram matrix (M x M)
            y: Labels (-1 or +1)
            alpha: Lagrange parameter

        Returns:
            Dual objective value.
        """
        return _dual_objective(G, y, alpha)

    def _step_length_selection(
        self,
        G: np.ndarray,
        y: np.ndarray,
        alpha_old: np.ndarray,
        alpha_new: np.ndarray,
    ) -> float:
        """
        Advanced Barzilai-Borwein step size using current and previous s, z values.

        Args:
            G: Gram matrix (M x M)
            y: Labels (-1 or +1)
            alpha_old: old Lagrange parameter
            alpha_new: new Lagrange parameter

        Returns:
            Step size.
        """
        if self._s_prev is not None and self._z_prev is not None:
            new_s, new_z, tau = _step_length_selection(
                self._s_prev,
                self._z_prev,
                G,
                y,
                alpha_old,
                alpha_new,
                self.tau_min,
                self.tau_max,
            )
            self._s_prev = new_s
            self._z_prev = new_z
            return tau

        s = alpha_new - alpha_old
        z = y * (G @ (y * s))  # Gradient difference approximation

        # Store current s and z for next iteration
        num = np.inner(s, s)
        denom = np.inner(s, z)

        self._s_prev = s
        self._z_prev = z

        tau = num / (denom + 1e-10)
        tau = np.clip(tau, self.tau_min, self.tau_max)

        return tau

    def _compute_gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the Gram matrix for the given data points based on the kernel.

        Args:
            X: Feature matrix (M x d).

        Returns:
            Gram matrix
        """
        G = self._kernel_function(
            X[:, None, ...], X[None, :, ...]
        )  # G: (M x M), X: (M x d)
        return G

    def _alpha_lambda(
        self, beta: np.ndarray, y: np.ndarray, lambd: float, C: float
    ) -> np.ndarray:
        """
        Compute alpha(lambda) based on the given formula (alpha(lambda) = min(max(beta + lambda * y, 0), C)).

        Args:
            beta: Lagrange parameter
            y: Labels (-1 or +1)
            lambd: Lagrange parameter
            C: Upper contstraint for the optimization problem, positive real

        Returns:
            Solution of the minimization of the partial Lagrangian.
        """
        return np.minimum(np.maximum(beta + lambd * y, 0), C)

    def _bracketing_phase(
        self,
        alpha: np.ndarray,
        y: np.ndarray,
        C: float,
        delta: float = 0.01,
        lambda_val: float = 0,
    ) -> Tuple[float, float, float, float, float]:
        """
        Perform the bracketing phase for bisection.

        Args:
            alpha: Lagrange parameter
            y: Labels (-1 or +1)
            C: Upper contstraint for the optimization problem, positive real
            delta: initial estimate
            lambda_val: initial value

        Returns: values to be used in the secant phase
            lambda_val: end value located in a bracket
            lambda_min: end value located in a bracket
            lambda_max: end value located in a bracket
            r_min: minimal value of single nonlinear equation
            r_max: maximal value of single nonlinear equation
        """
        alpha_projected = self._alpha_lambda(alpha, y, 0, C)

        r = np.inner(y, alpha_projected)

        # Bracketing phase
        if r < 0:
            lambda_min = lambda_val
            r_min = r
            lambda_val += delta
            alpha_lambda = self._alpha_lambda(alpha, y, lambda_val, C)
            r = np.inner(y, alpha_lambda)
            while r < 0:
                lambda_min = lambda_val
                r_min = r
                s = max(r_min / r - 1, 0.1)
                delta += delta / s
                lambda_val += delta
                alpha_lambda = self._alpha_lambda(alpha, y, lambda_val, C)
                r = np.inner(y, alpha_lambda)

            lambda_max = lambda_val
            r_max = r
        else:
            lambda_max = lambda_val
            r_max = r
            lambda_val -= delta
            alpha_lambda = self._alpha_lambda(alpha, y, lambda_val, C)
            r = np.inner(y, alpha_lambda)
            while r > 0:
                lambda_max = lambda_val
                r_max = r
                s = max(r_max / r - 1, 0.1)
                delta += delta / s
                lambda_val -= delta
                alpha_lambda = self._alpha_lambda(alpha, y, lambda_val, C)
                r = np.inner(y, alpha_lambda)

            lambda_min = lambda_val
            r_min = r
        return lambda_val, lambda_min, lambda_max, r_min, r_max

    def _secant_phase(
        self,
        alpha: np.ndarray,
        y: np.ndarray,
        C: float,
        delta: float,
        r_min: float,
        r_max: float,
        lambda_val: float,
        lambda_min: float,
        lambda_max: float,
    ) -> float:
        """
        Perform the secant phase for bisection.

        Args:
            alpha: Lagrange parameter
            y: Labels (-1 or +1)
            C: Upper contstraint for the optimization problem, positive real
            delta: initial estimate
            lambda_val: end value located in a bracket
            lambda_min: end value located in a bracket
            lambda_max: end value located in a bracket
            r_min: minimal value of single nonlinear equation (located in a bracket)
            r_max: maximal value of single nonlinear equation (located in a bracket)

        Returns:
            Lagrange parameter lambda.
        """
        s = 1 - r_min / (r_max + 1e-8)
        delta = delta / s
        lambda_val = lambda_max - delta
        r = np.inner(y, self._alpha_lambda(alpha, y, lambda_val, C))
        while abs(r) > self.tol:
            if r > 0:
                if s <= 2:
                    lambda_max = lambda_val
                    r_max = r
                    s = 1 - r_min / r_max
                    delta = (lambda_max - lambda_min) / (s + 1e-10)
                    lambda_val = lambda_max - delta
                else:
                    s = max(r_max / r - 1, 0.1)
                    delta = (lambda_max - lambda_val) / (s + 1e-10)
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
                    s = 1 - r_min / (r_max + 1e-8)
                    delta = (lambda_max - lambda_min) / (s + 1e-10)
                    lambda_val = lambda_max - delta
                else:
                    s = max(r_min / r - 1, 0.1)
                    delta = (lambda_val - lambda_min) / (s + 1e-10)
                    lambda_new = min(
                        lambda_val + delta, 0.75 * lambda_max + 0.25 * lambda_val
                    )
                    lambda_min = lambda_val
                    r_min = r
                    lambda_val = lambda_new
                    s = (lambda_max - lambda_min) / (lambda_max - lambda_val)

            r = np.inner(y, self._alpha_lambda(alpha, y, lambda_val, C))

        return lambda_val

    def _project_onto_feasible_set(
        self, alpha: np.ndarray, y: np.ndarray, C: float, delta: float = 0.01
    ) -> np.ndarray:
        """
        Project the alpha vector onto the feasible set using bisection method.

        Args:
            alpha: Lagrange parameter
            y: Labels (-1 or +1)
            C: Upper contstraint for the optimization problem, positive real
            delta: initial estimate

        Returns:
            Solution of the minimization of the partial Lagrangian.
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using the trained SVM.

        Args:
            X: Feature matrix (M x d).

        Returns:
            Values of +/- 1.
        """

        return np.sign(self._decision_function(X))

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values for given data points in both Primal and Dual SVM.

        Args:
            X: Feature matrix (M x d).

        Returns:
            Decision values, f(X) (M_test x 1)
        """
        if self.mode == "primal_SGD" or self.mode == "primal_QP":
            return np.dot(X, self.w) + self.b

        elif self.mode == "dual":
            sv = self.support_vectors[None, :, :]
            Xs = X[:, None, :]

            decision_values = (
                np.sum(
                    self.support_alphas
                    * self.support_y
                    * self._kernel_function(sv, Xs),
                    axis=-1,
                )
                + self.b
            )

            return decision_values

        else:
            raise ValueError("Mode must be 'primal_SGD', 'primal_QP' or 'dual'.")

    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Compute the kernel function between vectors x1 and x2.

        Args:
            x1: support vector
            x2: arbitrary element of the support vector, taking the first element for simplicity

        Returns:
        Computed kernel values.
        """

        if self.kernel == "linear":
            return np.einsum("...i,...i->...", x1, x2)

        elif self.kernel == "gaussian":
            return np.exp(-np.linalg.norm(x1 - x2, axis=-1) ** 2 / (2 * self.sigma**2))

        elif self.kernel == "laplacian":
            return np.exp(-np.linalg.norm(x1 - x2, axis=-1) / self.sigma)

        elif self.kernel == "inverse multiquadratic":
            return 1 / (self.sigma**2 + np.linalg.norm(x1 - x2, axis=-1) ** 2) ** self.s

        else:
            raise ValueError("Unsupported kernel function.")


from numba import njit


@njit
def _dual_objective(G: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> float:
    """
    Compute dual objective value.
    """
    return 0.5 * np.dot(alpha, y * (G @ (y * alpha))) - np.sum(alpha)


@njit
def _step_length_selection(
    s_prev: np.ndarray,
    z_prev: np.ndarray,
    G: np.ndarray,
    y: np.ndarray,
    alpha_old: np.ndarray,
    alpha_new: np.ndarray,
    tau_min: float,
    tau_max: float,
):
    """
    Advanced Barzilai-Borwein step size using current and previous s, z values.
    """
    s = alpha_new - alpha_old
    z = y * (G @ (y * s))  # Gradient difference approximation

    # Store current s and z for next iteration
    num = np.dot(s, s) + np.dot(s_prev, s_prev)
    denom = np.dot(s, z) + np.dot(z_prev, z_prev)

    tau = num / (denom + 1e-10)
    tau = min(max(tau, tau_min), tau_max)

    return s, z, tau


def main(
    w: np.ndarray, b: float, n_A: int, n_B: int, margin: float, kernel: str, mode: str
) -> None:
    """
    Training the SVM and predicting the decision boundary.

    Args:
        w: non-zero normal vector defining a hyperplane
        b: real number, offset of the hyperplane
        n_A: number of additional samples from class A
        n_B: number of additional samples from class B
        margin: desired margin for the samples
        kernel: function K: R^d x R^d -> R s.t. w(x) = inner product(w, K(x,))
        mode: primal (_SGD/_QP) or dual.

    Returns:
        None: Plot of the dataset and descision boundary.
    """
    import random
    from test_data import TestLinear
    import matplotlib.pyplot as plt

    random_seed = random.randint(0, 1000)

    listA, listB = TestLinear(w, b, n_A, n_B, margin, seed=random_seed)

    # Convert lists to numpy arrays
    X_A = np.array(listA)
    X_B = np.array(listB)
    X = np.vstack((X_A, X_B))
    y = np.hstack((np.ones(n_A), -np.ones(n_B)))  # Class A = +1, Class B = -1

    # Train the SVM
    svm = SVM(C=1.0, kernel=kernel, lr=0.01, mode=mode, sigma=1.5, s=1.0, max_iter=150)
    svm.fit(X, y)

    # Predict decision boundary
    xx, yy = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
    Z = np.c_[xx.ravel(), yy.ravel()]
    preds = svm.predict(Z).reshape(xx.shape)
    print("Finished predict")
    decision_values = svm._decision_function(Z).reshape(xx.shape)

    # Plot the results
    plt.figure(figsize=(9, 7))
    plt.contourf(
        xx,
        yy,
        decision_values,
        alpha=0.5,
        levels=[-100, 0, 100],
        colors=["#AFCBFF", "#F19C8A"],
    )
    plt.scatter(
        X_A[:, 0],
        X_A[:, 1],
        color="#FF5733",
        label="Class A",
        edgecolors="black",
        linewidth=0.5,
        alpha=0.85,
    )
    plt.scatter(
        X_B[:, 0],
        X_B[:, 1],
        color="#1F77B4",
        label="Class B",
        edgecolors="black",
        linewidth=0.5,
        alpha=0.85,
    )
    plt.legend(frameon=True, fontsize=12, loc="upper right")
    plt.title("SVM Decision Boundary on Generated Data")
    plt.show()


if __name__ == "__main__":
    main(np.array([1.0, 1.0]), 1.0, 2000, 2000, 0.5, "linear", "dual")
