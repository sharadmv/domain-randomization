import numpy as np

from dr.backend.base import Backend

class DartBackend(Backend):

    ENV_MAP = {
        'Hopper': 'DartHopper-v1',
        'Cheetah': 'DartHalfCheetah-v1',
        'Walker': 'DartWalker2d-v1',
    }

    def get_world(self, env):
        return env.env.dart_world

    def _get_limbs(self, env):
        return self.get_world(env).skeletons[1].bodynodes[2:]

    def get_masses(self, env):
        return np.array([b.mass() for b in self._get_limbs(env)])

    def set_masses(self, env, masses):
        for limb, mass in zip(self._get_limbs(env), masses):
            limb.set_mass(mass)

    def get_gravity(self, env):
        return -self.get_world(env).gravity()[1]

    def set_gravity(self, env, g):
        self.get_world(env).set_gravity([0, -g, 0])

    def get_restitutions(self, env):
        limbs = self._get_limbs(env)
        return self.get_world(env).gravity()[1]

    def get_damping_coefficients(self, env):
        limbs = self._get_limbs(env)
        return np.array([joint.damping_coefficient() for limb in limbs for joint in limb.parent_joint.dofs])

    def set_damping_coefficients(self, env, damping_coefficients):
        limbs = self._get_limbs(env)
        joints = [joint for limb in limbs for joint in limb.parent_joint.dofs]
        for joint, dc in zip(joints, damping_coefficients):
            joint.set_damping_coefficient(dc)
