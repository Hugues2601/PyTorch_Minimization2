import numpy as np
from scipy.integrate import simpson
import torch

# Ensure that CUDA is available
assert torch.cuda.is_available(), "CUDA is not available"

device = torch.device('cuda')


def heston_cf(phi, S0, T, r, kappa, v0, theta, sigma, rho):
    # Ensure that phi is a torch tensor on the GPU
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, dtype=torch.complex128, device=device)
    else:
        phi = phi.to(device).type(torch.complex128)

    # Ensure that S0, T, and r are tensors on the correct device and type
    # Avoid re-creating tensors for parameters that require gradients
    S0 = S0.to(device).type(torch.float64)
    T = T.to(device).type(torch.float64)
    r = r.to(device).type(torch.float64)
    # Parameters kappa, v0, theta, sigma, rho are already tensors with requires_grad=True

    a = -0.5 * phi ** 2 - 0.5 * 1j * phi
    b = kappa - rho * sigma * 1j * phi
    discriminant = torch.sqrt(b ** 2 - 2 * sigma ** 2 * a)
    g = (b - discriminant) / (b + discriminant)

    C = (
        kappa * (b - discriminant) / sigma ** 2 * T
        - 2 / sigma ** 2 * torch.log((1 - g * torch.exp(-discriminant * T)) / (1 - g))
    )
    D = (
        (b - discriminant) / sigma ** 2
        * (1 - torch.exp(-discriminant * T))
        / (1 - g * torch.exp(-discriminant * T))
    )

    cf = torch.exp(C * theta + D * v0 + 1j * phi * torch.log(S0 * torch.exp(r * T)))
    return cf



def heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho):
    # Ensure that S0, K, T, and r are tensors on the correct device and type
    # Avoid re-creating tensors for parameters that require gradients
    S0 = S0.to(device).type(torch.float64)
    K = K.to(device).type(torch.float64)
    T = T.to(device).type(torch.float64)
    r = r.to(device).type(torch.float64)
    # Parameters kappa, v0, theta, sigma, rho are already tensors with requires_grad=True

    params = (S0, T, r, kappa, v0, theta, sigma, rho)

    def integrand_P1(phi, K_i, T_i):
        # Define 1j as a torch tensor
        one_j = torch.tensor(1j, dtype=torch.complex128, device=device)
        # Ensure phi is a torch tensor
        phi = phi.to(device).type(torch.complex128)

        numerator = heston_cf(phi - one_j, S0, T_i, r, kappa, v0, theta, sigma, rho)
        denominator = heston_cf(-one_j, S0, T_i, r, kappa, v0, theta, sigma, rho) * (
            one_j * phi * K_i ** (one_j * phi)
        )
        return torch.real(numerator / denominator)

    def integrand_P2(phi, K_i, T_i):
        # Define 1j as a torch tensor
        one_j = torch.tensor(1j, dtype=torch.complex128, device=device)
        # Ensure phi is a torch tensor
        phi = phi.to(device).type(torch.complex128)

        numerator = heston_cf(phi, S0, T_i, r, kappa, v0, theta, sigma, rho)
        denominator = one_j * phi * K_i ** (one_j * phi)
        return torch.real(numerator / denominator)

    umax = 50  # upper limit for integration
    n_points = 500  # number of points for integration grid
    phi_values = torch.linspace(1e-5, umax, n_points, device=device)  # avoid phi=0 to prevent singularities

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
        weights = torch.ones(n_points, device=device)
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



def calibrate(S0, market_prices, K, T, r, initial_guess, epochs=30, lr=0.01):
    # Ensure that all inputs are torch tensors on the GPU
    S0 = torch.tensor(S0, dtype=torch.float64, device=device)
    K = torch.tensor(K, dtype=torch.float64, device=device)
    T = torch.tensor(T, dtype=torch.float64, device=device)
    market_prices = torch.tensor(market_prices, dtype=torch.float64, device=device)
    r = torch.tensor(r, dtype=torch.float64, device=device)

    # Initialize parameters to optimize
    kappa = torch.tensor(initial_guess['kappa'], dtype=torch.float64, device=device, requires_grad=True)
    v0 = torch.tensor(initial_guess['v0'], dtype=torch.float64, device=device, requires_grad=True)
    theta = torch.tensor(initial_guess['theta'], dtype=torch.float64, device=device, requires_grad=True)
    sigma = torch.tensor(initial_guess['sigma'], dtype=torch.float64, device=device, requires_grad=True)
    rho = torch.tensor(initial_guess['rho'], dtype=torch.float64, device=device, requires_grad=True)

    # Set up optimizer
    optimizer = torch.optim.Adam([kappa, v0, theta, sigma, rho], lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        # Compute model prices
        model_prices = heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho)
        # Compute loss (Mean Squared Error)
        loss = torch.mean((model_prices - market_prices) ** 2)
        # Backpropagation
        loss.backward()
        # Update parameters
        optimizer.step()
        # Print progress
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Return the calibrated parameters
    calibrated_params = {
        'kappa': kappa.item(),
        'v0': v0.item(),
        'theta': theta.item(),
        'sigma': sigma.item(),
        'rho': rho.item()
    }
    return calibrated_params

from datetime import datetime
import pandas as pd
def data_processing(df):

    df['expiration'] = pd.to_datetime(df['expiration'])

    # Calculer le temps restant en années
    today = datetime.today()
    df['TimeToMaturity'] = df['expiration'].apply(lambda x: (x - today).days / 365)
    mp = df.iloc[:, 0].tolist()
    T = df.iloc[:, -1].tolist()
    IV = df.iloc[:, 2].tolist()
    K = df.iloc[:, 3].tolist()

    return df, mp, T, IV, K

if __name__ == "__main__":
    # Example data
    df = pd.read_csv("C:\\Users\\hugue\\Desktop\\Master Thesis\\Data\\filtered_options_20252026.csv")
    file, market_prices, T, IV, K = data_processing(df)
    S0 = 207
    # market_prices = [10.0, 12.0, 8.0]  # Example market prices
    # K = [90.0, 100.0, 110.0]  # Corresponding strikes
    # T = [1.0, 1.0, 1.0]  # Corresponding maturities
    r = 0.05  # Risk-free rate

    # Initial guesses for parameters
    initial_guess = {
        'kappa': 2.0,
        'v0': 0.1,
        'theta': 0.1,
        'sigma': 0.2,
        'rho': -0.5
    }

    # Calibrate the model
    calibrated_params = calibrate(S0, market_prices, K, T, r, initial_guess, epochs=1000, lr=0.01)

    print("Calibrated Parameters:")
    print(calibrated_params)

