import numpy as np


def project_onto_feasible_set(self, gamma, y, C, delta_lambda=0.01):
    """
    Project the alpha vector onto the feasible set using bisection and bracketing phase.

    Args:
    - gamma: Current Lagrange multipliers (alpha values).
    - y: Class labels.
    - C: Regularization parameter.
    - delta_lambda: Initial step size for lambda.

    Returns:
    - Projected alpha values.
    """

    def compute_r(y, alpha_lambda):
        """Compute r = <y, alpha_lambda> to check feasibility."""
        return np.dot(y, alpha_lambda)

    # Step 1: Initial projection of gamma onto [0, C]
    alpha_projected = np.clip(gamma, 0, C)

    # Step 2: Check if constraint <y, alpha> = 0 is satisfied
    r = compute_r(y, alpha_projected)

    # If already feasible, return projection
    if abs(r) < self.tol:
        return alpha_projected

    # Step 3: Bracketing Phase
    lambda_val = 0
    lambda_l, lambda_u = None, None

    if r < 0:
        lambda_l = lambda_val
        lambda_val += delta_lambda
        alpha_lambda = np.clip(gamma + lambda_val * y, 0, C)
        r = compute_r(y, alpha_lambda)

        while r < 0:
            lambda_l = lambda_val
            s = max(abs(r) / abs(compute_r(y, alpha_lambda)) - 1, 0.1)
            delta_lambda += delta_lambda / s
            lambda_val += delta_lambda
            alpha_lambda = np.clip(gamma + lambda_val * y, 0, C)
            r = compute_r(y, alpha_lambda)

        lambda_u = lambda_val

    else:
        lambda_u = lambda_val
        lambda_val -= delta_lambda
        alpha_lambda = np.clip(gamma + lambda_val * y, 0, C)
        r = compute_r(y, alpha_lambda)

        while r > 0:
            lambda_u = lambda_val
            s = max(abs(r) / abs(compute_r(y, alpha_lambda)) - 1, 0.1)
            delta_lambda += delta_lambda / s
            lambda_val -= delta_lambda
            alpha_lambda = np.clip(gamma + lambda_val * y, 0, C)
            r = compute_r(y, alpha_lambda)

        lambda_l = lambda_val

    return alpha_lambda  # Return feasible projected alpha values
