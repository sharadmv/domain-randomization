from dr.backend.base import Backend

class DartBackend(Backend):

    ENV_MAP = {
        'Hopper': 'DartHopper-v1',
        'Cheetah': 'DartHalfCheetah-v1',
        'Walker': 'DartWalker2d-v1',
    }

    def get_world(self, env):
        return env.env.dart_world

    def get_masses(self, env):
        return [b.mass() for b in self.get_world(env).skeletons[1].bodynodes[2:]]

    def set_masses(self, env, masses):
        for limb, mass in zip(self.get_world(env).skeletons[1].bodynodes, masses):
            limb.set_mass(mass)

    def get_gravity(self, env):
        return -self.get_world(env).gravity()[1]

    def set_gravity(self, env, g):
        self.get_world(env).set_gravity([0, -g, 0])
