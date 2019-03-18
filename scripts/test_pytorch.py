import click

import numpy as np
from git import Repo
from datetime import datetime
import os.path as osp
import torch
from torch.multiprocessing import Pool
from pprint import pprint

repo = Repo('./')
branch = repo.active_branch.name


def test_torch_function(val, rank):
    tmp = torch.from_numpy(val)
    return tmp


@click.command()
@click.option('--experiment_name', type=str, default=osp.join(branch, datetime.now().strftime('%b%d_%H-%M-%S')))
@click.option('--env_name', type=str, default='Hopper')
@click.option('--backend', type=str, default='dart')
@click.option('--collision_detector', type=str, default='bullet')
@click.option('--num_timesteps', type=int, default=200000)
@click.option('--seed', type=int, default=0)
@click.option('--env_dist_stdev', type=float, default=0.0)
@click.option('--mean_scale', type=float, default=1.0)
@click.option('--pop_size', type=int, default=30)
@click.option('--num_elites', type=int, default=10)
@click.option('--debug', type=bool, default=False)
@click.option('--num_eval_env', type=int, default=100)
def main(experiment_name, env_name, backend, collision_detector,
         num_timesteps, seed, env_dist_stdev, mean_scale, pop_size, num_elites,
         debug, num_eval_env):

    val = np.random.normal(size=(4, 5), loc=0, scale=1)
    argss = [(val[rank], rank) for rank in range(4)]

    print('before')
    pprint(val)

    with Pool(4) as p:
        res = p.starmap(test_torch_function, argss)

    print('after')
    pprint(res)

if __name__ == "__main__":
    main()
