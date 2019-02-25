import numpy as np

from dr.backend.base import Backend

class MujocoBackend(Backend):

    ENV_MAP = {
        'Hopper': 'Hopper-v2',
        'Cheetah': 'HalfCheetah-v2',
        'Walker': 'Walker2d-v2',
    }

    def get_world(self, env):
        return env.env.model

    def get_masses(self, env):
        return self.get_world(env).body_mass[1:]

    def set_masses(self, env, masses):
        self.get_world(env).body_mass[1:] = masses

    def get_gravity(self, env):
        return -self.get_world(env).opt.gravity[-1]

    def set_gravity(self, env, g):
        self.get_world(env).opt.gravity[2] = -g
