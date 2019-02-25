import gym
from abc import abstractmethod, ABCMeta

ENVS = {
    'Hopper',
}

class Backend(object, metaclass=ABCMeta):

    def make(self, env_name):
        if env_name not in ENVS or env_name not in self.ENV_MAP:
            raise Exception(f"Cannot find environment {env_name}")
        env_name = self.ENV_MAP[env_name]
        return gym.make(env_name)
