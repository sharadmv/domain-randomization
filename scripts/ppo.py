import click
import dr

from git import Repo

repo = Repo('./')
branch = repo.active_branch.name

@click.command()
@click.option('--experiment_name', type=str, default=branch)
@click.option('--env_name', type=str, default='Hopper')
@click.option('--backend', type=str, default='dart')
@click.option('--collision_detector', type=str, default='bullet')
@click.option('--num_timesteps', type=int, default=1e6)
@click.option('--seed', type=int, default=0)
def main(experiment_name, env_name, backend, collision_detector, num_timesteps, seed):
    dr.experiment.PPO(
        experiment_name,
        dict(
            env_name=env_name,
            backend=backend,
            collision_detector=collision_detector
        ),
        num_timesteps, seed
    ).run()

if __name__ == "__main__":
    main()
