from efficient_fpt.models import DDModel, piecewise_const_func
from efficient_fpt.utils import get_alternating_addm_mu_array

class aDDModel(DDModel):
    """
    One trial of an attentional drift diffusion model with alternating drift mu1 and mu2.
    """
    def __init__(self, mu1, mu2, sacc_array, flag, sigma, a, b, x0):
        super().__init__(x0)
        # drift parameters
        self.mu1 = mu1
        self.mu2 = mu2
        self.sacc_array = sacc_array
        self.flag = flag # indicates whether the process starts with mu1 (flag=0) or mu2 (flag=1)
        self.d = len(sacc_array) # number of stages
        self.mu_array = get_alternating_addm_mu_array(mu1, mu2, self.d, flag)
        # diffusion parameter
        self.sigma = sigma
        # symmetric linear boundary parameters
        self.a = a
        self.b = b

    def drift_coeff(self, X: float, t: float) -> float:
        return piecewise_const_func(t, self.mu_array, self.sacc_array)

    def diffusion_coeff(self, X: float, t: float) -> float:
        return self.sigma

    @property
    def is_update_vectorizable(self) -> bool:
        return True

    def upper_bdy(self, t: float) -> float:
        return self.a - self.b * t

    def lower_bdy(self, t: float) -> float:
        return -self.a + self.b * t