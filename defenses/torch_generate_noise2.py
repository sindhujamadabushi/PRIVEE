import numpy as np
import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Gaussian:
    def __init__(self, rho, random_state=None):
            self.rho = rho

            self._scale = 0.48448 / self.rho
        #   self._scale = 0.2 / self.epsilon
            self._rng = np.random.RandomState()

    
    def randomise(self, data, bound: float = 3.0) -> np.ndarray:
        
        if data.ndim == 1:
            num_classes = data.shape[0]
        else:
            num_classes = data.shape[1]

        edges = np.linspace(0, 1, num_classes + 1)
        
        noise = np.array([
            self._rng.uniform(low=edges[i], high=edges[i+1])
            for i in range(num_classes)
        ])

        rho_parts = np.linspace(0.05, 0.1, num_classes + 1)[1:]

        base = 0.48448
        scales = base / rho_parts   # shape = (num_classes,s)
        
        std_scaling = noise * scales

        noise_matrix = np.diag(std_scaling)
        
        noisy_data = data + noise_matrix
        
        return noisy_data
  
def add_Gaussian_noise_priveeplus(confidence_scores, rho):
    
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
