import tensorflow as tf
from baselines.common.cmd_util import common_arg_parser
from baselines.common import tf_util as U
from baselines import logger

import dr
import numpy as np
import random


def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def train(env_id, backend, num_timesteps, seed, stdev=0., collision_detector='bullet'):
    tf.set_random_seed(seed)
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env_dist = dr.dist.Normal(env_id, backend, stdev=stdev)
    env_dist.seed(seed)
    env = env_dist.sample()
    env.seed(seed)
    env_dist.backend.set_collision_detector(env, collision_detector)
    set_global_seeds(seed)

    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()

def parse_args():
    argparser = common_arg_parser()
    argparser.add_argument('--backend', type=str, default='dart')
    argparser.add_argument('--collision_detector', type=str, default='bullet')
    argparser.set_defaults(env='Hopper')
    return argparser.parse_args()

def main():
    args = parse_args()
    logger.configure()
    train(args.env, args.backend, num_timesteps=args.num_timesteps, seed=args.seed,
          collision_detector=args.collision_detector)

if __name__ == '__main__':
    main()
