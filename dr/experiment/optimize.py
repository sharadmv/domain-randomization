from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import git
import random

import scipy.stats as stats
import numpy as np
import tensorflow as tf
from baselines.common import tf_util as U
from baselines import logger
from parasol.experiment import Experiment

import dr
from datetime import datetime
import pickle
from path import Path
import tensorboardX
import sys
from mpi4py import MPI
COMM = MPI.COMM_WORLD


def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


class CEMOptimizer(object):

    def __init__(self, sol_dim, max_iters, popsize, num_elites, cost_function,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites

        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        self.cost_function = cost_function

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        mean, var, t = init_mean, init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
            samples = samples.astype(np.float32)

            costs = self.cost_function(samples, t)

            elites = samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1

        return mean


class Optimize(Experiment):

    def __init__(self, experiment_name, env_params, train_params, **kwargs):
        self.env_params = env_params
        self.train_params = train_params
        self.optimizer = CEMOptimizer(
            sol_dim=18,
            max_iters=300,
            popsize=train_params['pop_size'],
            num_elites=train_params['num_elites'],
            cost_function=self._cost_function,
            lower_bound=0.0,
        )
        super().__init__(experiment_name, **kwargs)

    def from_dict(self, params):
        return Optimize(params['experiment_name'], params['env_params'],
                        params['train_params'])

    def initialize(self, out_dir):
        pass

    def to_dict(self):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return {
            'env_params': self.env_params,
            'train_params': self.train_params,
            'git_hash': sha,
            'experiment_name': self.experiment_name,
        }

    def train(self, env_id, backend, num_timesteps, seed, viz_logdir, means, stdevs,
              collision_detector='bullet', rank=None, env_dist=None):

        from baselines.ppo1 import mlp_policy, pposgd_simple

        def policy_fn(name, ob_space, ac_space):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                        hid_size=64, num_hid_layers=2)

        assert len(means) == len(stdevs), (len(means), len(stdevs))
        mean_dict, stdev_dict = Optimize._vec_to_dict(env_id, means, stdevs)

        if hasattr(self, 'env_dist'):
            self.env_dist.default_parameters = mean_dict
            self.env_dist.stdev_dict = stdev_dict
        else:
            self.env_dist = dr.dist.Normal(env_id, backend, mean_dict=mean_dict, stdev_dict=stdev_dict)

        self.env_dist.seed(seed)

        # tf_config = tf.ConfigProto(
        #     allow_soft_placement=True,
        #     inter_op_parallelism_threads=1,
        #     intra_op_parallelism_threads=1)
        #
        # with tf.Session(config=tf_config, graph=tf.get_default_graph()) as sess:
        pi, eval_perfs = pposgd_simple.learn(self.env_dist, collision_detector, policy_fn,
                max_timesteps=num_timesteps,
                timesteps_per_actorbatch=2048,
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                gamma=0.99, lam=0.95, schedule='linear', viz_logdir=viz_logdir, eval_envs=self.eval_envs,
                eval_freq=2000
            )
            # sess.close()

        tf.reset_default_graph()

        return eval_perfs

    @classmethod
    def _vec_to_dict(cls, env_id, means, stdevs):

        assert env_id == 'Hopper', 'Only support Hopper for now'

        return dict(
            mass=means[:4],
            damping=means[4:-1],
            gravity=means[-1]
        ), dict(
            mass=stdevs[:4],
            damping=stdevs[4:-1],
            gravity=stdevs[-1]
        )

    @classmethod
    def _dict_to_vec(cls, env_id, d):

        assert env_id == 'Hopper', 'Only support Hopper for now'

        return np.concatenate((
            d['mass'],
            d['damping'],
            [d['gravity']]
        )).flatten().copy()

    def _cost_function(self, samples, cem_timestep):

        env_name = self.env_params['env_name']
        backend = self.env_params['backend']
        collision_detector = self.env_params['collision_detector']

        num_ts = self.train_params['num_timesteps']
        seed = self.train_params['seed']

        viz_logdir = '/tmp'

        argss = [(env_name, backend, num_ts, seed, viz_logdir, samples[rank][:len(samples[rank])//2],
                 samples[rank][len(samples[rank])//2:], collision_detector, rank) for rank in range(len(samples))]

        print('BEFORE SHIT WENT DOWN')

        # Send args to other MPI processes
        for rank in range(1, COMM.size):
            COMM.send(argss[rank], dest=rank)

        # Obtain results for all args
        r = self.train(*argss[0])

        reses = [(0, r)] # 0 is the rank of this process

        # Receive results from the other processes:
        for rank in range(1, COMM.size):
            r = COMM.recv(source=rank)
            reses.append((rank, r))

        reses = sorted(reses, key=lambda k: k[0])
        print(reses)

        # Get the index of the highest performing model in population
        # and write result to tensorboard
        max_idx = 0
        max_perf = max(reses[0][1]) # 0 is the result of process rank 0. 1 brings us the eval perf list

        for i, item in enumerate(reses):
            perf = max(item[1])
            if perf > max_perf:
                max_perf = perf
                max_idx = i

        self.writer.add_scalar('max_perf', max_perf, cem_timestep)

        with open(self.out_dir / f'eval_perfs_{max_idx}.pkl', 'wb') as f:
            pickle.dump(reses[max_idx], f)

        # Obtain the "costs" that the CEM cost function should return
        costs = [- max(i[1]) for i in reses]

        print(costs)

        return costs

    def run_experiment(self, out_dir):
        logger.configure()

        set_global_seeds(self.train_params['seed'])

        viz_logdir = 'runs/' + str(self.env_params) + str(self.train_params) + datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = tensorboardX.SummaryWriter(log_dir=viz_logdir)

        self.out_dir = Path(out_dir)

        # Obtain initial mean and std
        env_name = self.env_params['env_name']
        backend = self.env_params['backend']

        stdev = self.train_params['env_dist_stdev']
        mean_scale = self.train_params['mean_scale']

        env_dist = dr.dist.Normal(env_name, backend, mean_scale=mean_scale)
        init_mean_param = Optimize._dict_to_vec(env_name, env_dist.default_parameters)

        init_stdev_param = np.array([stdev] * len(init_mean_param), dtype=np.float32)

        cem_init_mean = np.concatenate((init_mean_param, init_stdev_param))
        cem_init_stdev = np.array([1.0] * len(cem_init_mean), dtype=np.float32)

        # initialize upper bound for cem optimizer
        self.optimizer.ub = cem_init_mean * 5.0

        # Creating eval_envs here
        eval_envs = [env_dist.backend.make(env_dist.env_name) for _ in range(self.train_params['num_eval_env'])]
        [env_dist.backend.set_collision_detector(e, self.env_params['collision_detector']) for e in eval_envs]
        self.eval_envs = eval_envs

        if COMM.Get_rank() == 0:
            self.optimizer.obtain_solution(cem_init_mean, cem_init_stdev)

            COMM.Abort(0)
        else:
            while True:
                args = COMM.recv(source=0)

                r = self.train(*args)

                COMM.send(r, dest=0)


