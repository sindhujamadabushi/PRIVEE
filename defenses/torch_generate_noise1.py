import numpy as np
import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Gaussian:
    def __init__(self, rho, random_state=None):
            self._rho = rho
            self._scale = 0.48448 / self._rho
            self._rng = np.random.RandomState()
    
    def randomise(self, data, bound: float = 3.0) -> np.ndarray:
        # self._check_all()
        num_classes = len(data)
        
        edges = np.linspace(0, 3, num_classes + 1)  # length = num_classes+1
        
        noise = np.array([
            self._rng.uniform(low=edges[i], high=edges[i+1])
            for i in range(num_classes)
        ])
        std_scaling = noise * self._scale
        noise_matrix = np.diag(std_scaling)
        noisy_data = data + noise_matrix
        return noisy_data
    
def add_Gaussian_noise_privee(confidence_scores, rho):
    """
    Vectorized implementation: sort each row, apply Apert, then scatter back.

    Inputs:
      - confidence_scores: torch.Tensor of shape (N, K)
      - epsilon, delta, sensitivity: Gaussian mechanism parameters
    Returns:
      - torch.Tensor of shape (N, K) on the same device, dtype float32
    """
    # 1) Move to CPU and convert to NumPy
    conf_np = confidence_scores.detach().cpu().numpy()  # shape (N, K)
    N, K = conf_np.shape

    # 2) Build A and Apert
    A = (-2.0 / K) * np.ones((K, K)) + np.eye(K)
    mechanism = Gaussian(rho=rho)
    Apert = mechanism.randomise(A)  # shape (K, K)

    # 3) Compute sort indices and sorted values (axis=1 sorts each row)
    sort_idx = np.argsort(conf_np, axis=1)                     # shape (N, K)
    sorted_vals = np.take_along_axis(conf_np, sort_idx, axis=1)  # shape (N, K)

    # 4) Apply Apert to each sorted row: (N, K) dot (K, K).T → (N, K)
    Upert_sorted = sorted_vals.dot(Apert.T)  # shape (N, K)

    # 5) Scatter perturbed values back into original order
    Upert_np = np.empty_like(Upert_sorted)
    rows = np.arange(N)[:, None]              # shape (N, 1)
    Upert_np[rows, sort_idx] = Upert_sorted   # scatter per-row

    # 6) Convert back to torch.FloatTensor on original device
    return torch.from_numpy(Upert_np.astype(np.float32)).to(confidence_scores.device)
