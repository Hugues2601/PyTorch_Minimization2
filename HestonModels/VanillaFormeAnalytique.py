import torch
from config import *

def heston_cf(phi, S0, T, r, kappa, v0, theta, sigma, rho):
    # Ensure that phi is a torch tensor on the GPU
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, dtype=torch.complex128, device=CONFIG.device)
    else:
        phi = phi.to(CONFIG.device).type(torch.complex128)

    # Ensure that S0, T, and r are tensors on the correct CONFIG.device and type
    # Avoid re-creating tensors for parameters that require gradients
    S0 = S0.to(CONFIG.device).type(torch.float64)
    T = T.to(CONFIG.device).type(torch.float64)
    r = r.to(CONFIG.device).type(torch.float64)
    # Parameters kappa, v0, theta, sigma, rho are already tensors with requires_grad=True

    a = -0.5 * phi ** 2 - 0.5 * 1j * phi
    b = kappa - rho * sigma * 1j * phi
    discriminant = torch.sqrt(b ** 2 - 2 * sigma ** 2 * a)
    g = (b - discriminant) / (b + discriminant)

    C = (kappa * (b - discriminant) / sigma ** 2 * T- 2 / sigma ** 2 * torch.log((1 - g * torch.exp(-discriminant * T)) / (1 - g)))
    D = ((b - discriminant) / sigma ** 2 * (1 - torch.exp(-discriminant * T)) / (1 - g * torch.exp(-discriminant * T)))

    cf = torch.exp(C * theta + D * v0 + 1j * phi * torch.log(S0 * torch.exp(r * T)))
    return cf




def heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho):
    # Ensure that S0, K, T, and r are tensors on the correct CONFIG.device and type
    # Avoid re-creating tensors for parameters that require gradients
    S0 = S0.to(CONFIG.device).type(torch.float64)
    K = K.to(CONFIG.device).type(torch.float64)
    T = T.to(CONFIG.device).type(torch.float64)
    r = r.to(CONFIG.device).type(torch.float64)
    # Parameters kappa, v0, theta, sigma, rho are already tensors with requires_grad=True

    params = (S0, T, r, kappa, v0, theta, sigma, rho)

    def integrand_P1(phi, K_i, T_i):
        # Define 1j as a torch tensor
        one_j = torch.tensor(1j, dtype=torch.complex128, device=CONFIG.device)
        # Ensure phi is a torch tensor
        phi = phi.to(CONFIG.device).type(torch.complex128)

        numerator = heston_cf(phi - one_j, S0, T_i, r, kappa, v0, theta, sigma, rho)
        denominator = heston_cf(-one_j, S0, T_i, r, kappa, v0, theta, sigma, rho) * (
            one_j * phi * K_i ** (one_j * phi)
        )
        return torch.real(numerator / denominator)

    def integrand_P2(phi, K_i, T_i):
        # Define 1j as a torch tensor
        one_j = torch.tensor(1j, dtype=torch.complex128, device=CONFIG.device)
        # Ensure phi is a torch tensor
        phi = phi.to(CONFIG.device).type(torch.complex128)

        numerator = heston_cf(phi, S0, T_i, r, kappa, v0, theta, sigma, rho)
        denominator = one_j * phi * K_i ** (one_j * phi)
        return torch.real(numerator / denominator)

    umax = 50  # upper limit for integration
    n_points = 500  # number of points for integration grid
    phi_values = torch.linspace(1e-5, umax, n_points, device=CONFIG.device)  # avoid phi=0 to prevent singularities

    # Initialize lists to store P1 and P2 values
    P1_list = []
    P2_list = []
    for i in range(len(K)):
        K_i = K[i]
        T_i = T[i]

        integrand_P1_values = integrand_P1(phi_values, K_i, T_i)
        integrand_P2_values = integrand_P2(phi_values, K_i, T_i)

        # Implement Simpson's rule in PyTorch
        dx = (umax - 1e-5) / (n_points - 1)

        # Simpson's rule coefficients
        weights = torch.ones(n_points, device=CONFIG.device)
        weights[1:-1:2] = 4
        weights[2:-2:2] = 2
        weights = weights * dx / 3

        integral_P1 = torch.sum(weights * integrand_P1_values)
        integral_P2 = torch.sum(weights * integrand_P2_values)

        P1 = 0.5 + 1 / torch.pi * integral_P1
        P2 = 0.5 + 1 / torch.pi * integral_P2

        P1_list.append(P1)
        P2_list.append(P2)

    P1 = torch.stack(P1_list)
    P2 = torch.stack(P2_list)

    price = S0 * P1 - torch.exp(-r * T) * K * P2
    return price