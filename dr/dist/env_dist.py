import numpy as np
from abc import abstractmethod, ABCMeta

from dr.backend import get_backend
from dr.envs import get_driver

class EnvironmentDistribution(object, metaclass=ABCMeta):

    def __init__(self, env_name, backend_name):
        self.env_name, self.backend_name = env_name, backend_name
        self.backend = get_backend(backend_name)
        self.env_driver = get_driver(env_name, self.backend)
        self.root_env = self.backend.make(self.env_name)
        self.root_env.env.disableViewer = False
        self.default_parameters = self.env_driver.get_parameters(self.root_env)
        self._seed = None

    def sample(self, in_place=True):
        if self.in_place:
            env = self.root_env
        else:
            env = self.backend.make(self.env_name)
            env.env.disableViewer = False
        parameters = self.env_driver.get_parameters(env)
        parameters = self._sample(parameters)
        self.env_driver.set_parameters(env, parameters)
        return env

    @abstractmethod
    def _sample(self, parameters):
        pass

    def seed(self, seed):
        np.random.seed(seed)
        self._seed = seed
