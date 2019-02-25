import numpy as np

from dr.envs.driver import Driver

class Walker(Driver):

    def get_parameters(self, env):
        return np.concatenate([self.backend.get_masses(env), [self.backend.get_gravity(env)]])

    def set_parameters(self, env, parameters):
        masses = parameters[:-1]
        gravity = parameters[-1]
        self.backend.set_masses(env, masses)
        self.backend.set_gravity(env, gravity)
