import numpy as np
import dr

dist = dr.dist.Normal('Walker', 'mujoco', stdev=0.1)

def sample_action():
    return np.random.uniform(low=env.action_space.low, high=env.action_space.high)

dist.seed(0)
env = dist.sample()
env.seed(0)
np.random.seed(0)
env.reset()
env.render()
for _ in range(1000):
    action = sample_action()
    state, _, _, _ = env.step(action)
    env.render()

dist.seed(1)
env = dist.sample()
env.seed(0)
np.random.seed(0)
env.reset()
env.render()
for _ in range(1000):
    action = sample_action()
    state, _, _, _ = env.step(action)
    env.render()
