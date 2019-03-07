import git
import random

import tensorflow as tf
import numpy as np
from baselines.common.cmd_util import common_arg_parser
from baselines.common import tf_util as U
from baselines import logger
gfile = tf.gfile
from parasol.experiment import Experiment

import dr
from datetime import datetime


def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


class PPO(Experiment):

    def __init__(self, experiment_name, env_params, train_params, **kwargs):
        self.env_params = env_params
        self.train_params = train_params
        super(PPO, self).__init__(experiment_name, **kwargs)

    def from_dict(self, params):
        return PPO(params['env'], params['num_timesteps'],
                   params['seed'])

    def initialize(self, out_dir):
        pass

    def to_dict(self):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return {
            'env_params': self.env_params,
            'train_params': self.train_params,
            'git_hash': sha
        }

    def train(self, env_id, backend, num_timesteps, seed, viz_logdir, stdev=0.,
              collision_detector='bullet'):
        from baselines.ppo1 import mlp_policy, pposgd_simple
        U.make_session(num_cpu=1).__enter__()
        def policy_fn(name, ob_space, ac_space):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                hid_size=64, num_hid_layers=2)
        env_dist = dr.dist.Normal(env_id, backend, stdev=stdev)
        env_dist.seed(seed)
        set_global_seeds(seed)

        pposgd_simple.learn(env_dist, collision_detector, policy_fn,
                max_timesteps=num_timesteps,
                timesteps_per_actorbatch=2048,
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                gamma=0.99, lam=0.95, schedule='linear', viz_logdir=viz_logdir
            )

    def run_experiment(self, out_dir):
        logger.configure()

        env_name = self.env_params['env_name']
        backend = self.env_params['backend']
        collision_detector = self.env_params['collision_detector']

        num_ts = self.train_params['num_timesteps']
        seed = self.train_params['seed']
        stdev = self.train_params['env_dist_stdev']

        viz_logdir = 'runs/' + str(self.env_params) + str(self.train_params) + datetime.now().strftime('%b%d_%H-%M-%S')

        self.train(env_name, backend, num_timesteps=num_ts, seed=seed, viz_logdir=viz_logdir,
                   collision_detector=collision_detector, stdev=stdev, )
