import gym

from dr.backend.base import Backend, ENVS


class DartBackend(Backend):

    ENV_MAP = {
        'Hopper': 'DartHopper-v1'
    }
