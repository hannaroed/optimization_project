import numpy as np

def projected_gradient_descent(alpha_initial: float, d_intial: float, tau: float, f: function,  error: float = 10e-6, max_iter: int = 1000):
    alpha = alpha_initial
    d = d_intial
    iter = 0
    diff = 1
    while diff > error:
        d_new = projection(alpha - tau * np.gradient(f(alpha))) - alpha
        alpha_new = alpha + d_new
        tau = step_length_selection(alpha, alpha_new)
        diff = np.linalg.norm(alpha_new - alpha)
        alpha = alpha_new
        d = d_new
        iter += 1
        if iter > max_iter:
            break
    return alpha

def projection():
    pass

def step_length_selection(alpha: float, alpha_new: float):
    tau_min = 10e-5
    tau_max = 10e5
    s = alpha_new - alpha
    z = np.gradient(alpha_new) - np.gradient(alpha)
    if np.dot(s, z) <= 0: 
        return tau_max
    else:
        tau = np.dot(s, s) / np.dot(s, z)
        tau = max(tau_min, min(tau, tau_max))
    return tau