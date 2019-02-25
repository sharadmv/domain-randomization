import numpy as np
from dr.dist.env_dist import EnvironmentDistribution

class Normal(EnvironmentDistribution):

    def __init__(self, *args, **kwargs):
        self.scale = kwargs.pop('stdev', 1.0)
        super(Normal, self).__init__(*args, **kwargs)

    def _sample(self, params):
        params = np.random.normal(params, scale=self.scale)
        print(params)
        return params
