from abc import abstractmethod, ABCMeta

from dr.backend import get_backend

class EnvironmentDistribution(object, metaclass=ABCMeta):

    def __init__(self, env_name, backend_name):
        self.env_name, self.backend_name = env_name, backend_name
        self.backend = get_backend(backend_name)

    def sample(self):
        pass
