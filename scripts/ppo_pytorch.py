import click
import dr

from git import Repo
from datetime import datetime
import os.path as osp

repo = Repo('./')
branch = repo.active_branch.name

@click.command()
@click.option('--experiment_name', type=str, default=osp.join(branch, datetime.now().strftime('%b%d_%H-%M-%S')))
@click.option('--env_name', type=str, default='Hopper')
@click.option('--backend', type=str, default='dart')
@click.option('--collision_detector', type=str, default='bullet')
@click.option('--num_timesteps', type=int, default=1e6)
@click.option('--seed', type=int, default=0)
@click.option('--env_dist_stdev', type=float, default=0.0)
@click.option('--mean_scale', type=float, default=1.0)
def main(experiment_name, env_name, backend, collision_detector,
         num_timesteps, seed, env_dist_stdev, mean_scale):

    assert env_dist_stdev == 0.0
    assert mean_scale == 1.0

    dr.experiment.PPO_Pytorch(
        experiment_name,
        env_params = dict(
            env_name=env_name,
            backend=backend,
            collision_detector=collision_detector
        ),
        train_params = dict(
            num_timesteps=num_timesteps,
            seed=seed,
            env_dist_stdev=env_dist_stdev,
            mean_scale=mean_scale,

            #PPO parameter,
            hid_size=64,
            pol_init_std=1.0,
            adam_epsilon=1e-5,
            optim_stepsize=3e-4,

            ts_per_batch = 2048,
            lam = 0.95,
            gamma = 0.99,
            optim_epoch = 10,
            optim_batch_size = 64,
            clip_param = 0.2

        )
    ).run()


if __name__ == "__main__":
    main()
