import gym
from abc import abstractmethod, ABCMeta

class Backend(object, metaclass=ABCMeta):

    def make(self, env_name):
        if env_name not in self.ENV_MAP:
            raise Exception(f"Cannot find environment {env_name}")
        env_name = self.ENV_MAP[env_name]
        return gym.make(env_name)

    @abstractmethod
    def get_world(self, env):
        pass

    @abstractmethod
    def get_masses(self, env):
        pass

    @abstractmethod
    def set_masses(self, env, masses):
        pass

    @abstractmethod
    def set_gravity(self, env, g):
        pass
