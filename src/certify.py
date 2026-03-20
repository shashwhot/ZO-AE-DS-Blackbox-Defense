import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint # type: ignore
from typing import Tuple

class Smooth(object):
    """
    The Randomized Smoothing wrapper. It takes a base classifier and evaluates 
    its robustness by bombarding inputs with Gaussian noise.
    """
    ABSTAIN = -1 # Returned if the model is too confused to make a safe guess

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma # The intensity of the Gaussian noise

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> Tuple[int, float]:
        """
        Calculates the 'Certified Radius'. This guarantees the model's prediction 
        cannot be changed by an attacker within a certain L2 distance.
        """
        self.base_classifier.eval()
        
        # Step 1: Take a small number of samples (n0) to guess the most likely class
        counts_selection = self._sample_noise(x, n0, batch_size)
        cAHat = counts_selection.argmax().item()
        
        # Step 2: Take a massive number of samples (n) to rigorously test that guess
        counts_estimation = self._sample_noise(x, n, batch_size)
        nA = counts_estimation[cAHat].item()
        
        # Step 3: Calculate the lower confidence bound using statistics
        pABar = self._lower_confidence_bound(nA, n, alpha)

        # Step 4: If confidence is too low (< 50%), abstain. Otherwise, calculate the safe radius.
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Standard prediction using majority vote over noisy samples. """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1, count2 = counts[top2[0]], counts[top2[1]]
        
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Helper function: Creates noisy copies of the image and passes them to the model. """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                # Duplicate the image 'batch_size' times
                batch = x.repeat((this_batch_size, 1, 1, 1))
                # Generate Gaussian noise
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                # Ask the model what it sees in the noisy images
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Statistical helper: Clopper-Pearson confidence interval """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]