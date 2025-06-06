import numpy as np
import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Gaussian:
  def __init__(self, epsilon, delta, sensitivity, random_state=None):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
      #   self._scale = 0.2 / self.epsilon
        self._rng = np.random.RandomState()

  def _check_epsilon_delta(cls, epsilon, delta):
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")

        if epsilon > 1.0:
            raise ValueError("Epsilon cannot be greater than 1. If required, use GaussianAnalytic instead.")
        return super()._check_epsilon_delta(epsilon, delta)

  def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, float):
            raise TypeError("Sensitivity must be numeric")
        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")
        return float(sensitivity)

  def _check_all(self):
        self._check_sensitivity(self.sensitivity)
        return True

  def bias(self):
        return 0.0

  def variance(self):
        self._check_all(0)
        return self._scale ** 2
  
  def randomise(self, data):
        self._check_all()
        standard_normal = (self._rng.standard_normal(data.shape) + self._rng.standard_normal(data.shape)) / np.sqrt(2)
        standard_normal_tensor = torch.tensor(standard_normal)
        data = data.to(device)
        std_scaling = standard_normal_tensor * self._scale
        std_scaling = std_scaling.to(device)
        noisy_data = data + std_scaling
        noisy_data = noisy_data.to(device)
        return noisy_data
  
def add_Gaussian_noise_raw(data, epsilon, delta, sensitivity):
    mechanism = Gaussian(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
    noisy_data = mechanism.randomise(data)
    noisy_data = noisy_data.to(device)
    return noisy_data
  




