import random

import numpy as np

import dr
from datetime import datetime
import pickle
from path import Path
import torch

from dr.ppo.utils import set_torch_num_threads, RunningMeanStd, traj_seg_gen
from dr.ppo.train import one_train_iter
from dr.ppo.models import Policy, ValueNet
import torch.optim as optim
from collections import deque
from tqdm import trange


def set_global_seeds(i):
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(i)


class PPO_Pytorch(object):

    def __init__(self, experiment_name, env_params, train_params, **kwargs):
        self.experiment_name = experiment_name
        self.env_params = env_params
        self.train_params = train_params
        super().__init__()

    def train(self, env_id, backend,
              train_params,
              seed, viz_logdir,
              stdev=0., mean_scale=1.0, collision_detector='bullet'):

        # Unpack params
        hid_size = train_params['hid_size']
        pol_init_std = train_params['pol_init_std']
        adam_epsilon = train_params['adam_epsilon']
        optim_stepsize = train_params['optim_stepsize']

        # Make env
        env_dist = dr.dist.Normal(env_id, backend, stdev=stdev, mean_scale=mean_scale)
        env_dist.backend.set_collision_detector(env_dist.root_env, collision_detector)
        env_dist.seed(seed)
        env = env_dist.root_env

        eval_envs = [env_dist.backend.make(env_dist.env_name) for _ in range(100)]
        [env_dist.backend.set_collision_detector(e, collision_detector) for e in eval_envs]

        set_torch_num_threads()

        # Construct policy and value network
        pol = Policy(env.observation_space, env.action_space, hid_size, pol_init_std)
        pol_optim = optim.Adam(pol.parameters(), lr=optim_stepsize, eps=adam_epsilon)

        val = ValueNet(env.observation_space, hid_size)
        val_optim = optim.Adam(val.parameters(), lr=optim_stepsize, eps=adam_epsilon)

        optims = {'pol_optim': pol_optim, 'val_optim': val_optim}

        num_train_iter = int(train_params['num_timesteps'] / train_params['ts_per_batch'])

        # Buffer for running statistics
        eps_rets_buff = deque(maxlen=100)
        eps_rets_mean_buff = []

        state_running_m_std = RunningMeanStd(shape=env.observation_space.shape)

        # seg_gen is a generator that yields the training data points
        seg_gen = traj_seg_gen(env, pol, val, state_running_m_std, train_params)

        eval_perfs = []

        for iter_i in trange(num_train_iter):
            print('\nStarting training iter', iter_i)
            one_train_iter(pol, val, optims,
                           iter_i, eps_rets_buff, eps_rets_mean_buff, seg_gen,
                           state_running_m_std, train_params, eval_envs, eval_perfs)
            print()

    def run(self):

        env_name = self.env_params['env_name']
        backend = self.env_params['backend']
        collision_detector = self.env_params['collision_detector']

        seed = self.train_params['seed']
        stdev = self.train_params['env_dist_stdev']
        mean_scale = self.train_params['mean_scale']

        set_global_seeds(seed)

        viz_logdir = 'runs/' + str(self.env_params) + str(self.train_params) + datetime.now().strftime('%b%d_%H-%M-%S')

        self.train(
            env_id=env_name,
            backend=backend,
            train_params=self.train_params,
            seed=seed,
            viz_logdir=viz_logdir,
        )
