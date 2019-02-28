import numpy as np
import scipy.stats as stats
from dr.dist.env_dist import EnvironmentDistribution


class Normal(EnvironmentDistribution):

    def __init__(self, *args, **kwargs):
        self.scale = kwargs.pop('stdev', 1.0)
        super(Normal, self).__init__(*args, **kwargs)

    def _sample(self, params):

        # Sample gravity parameter
        g = np.random.normal(params[-1], scale=self.scale)

        # Mass parameters can only be positive,
        # We thus sample them from a truncated normal distribution
        # We set the lower bound to an arbitrary small positive number
        # since stats.truncnorm samples in closed interval [a, b].
        # The distribution is theoretically un-bounded from above
        # But we set the upper bound to be an arbitrary large number
        # to avoid any potential numerical issue with setting upper bound to be +inf
        mass_means = params[:-1]

        X = stats.truncnorm(1e-7, 1e7, loc=mass_means, scale=self.scale)
        mass = X.rvs(size=len(mass_means))

        sample = np.concatenate((mass, [g]))

        assert len(sample) == len(params)

        return sample
