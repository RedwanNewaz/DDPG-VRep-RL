from task import PathFollow
from models import actor_critic
from examples import vertical_hover
import numpy as np

np.random.seed(123)
class full_control(vertical_hover):
    print('full control class initialized')





if __name__ == '__main__':
    env = PathFollow(headless=False)
    env.seed(123)
    policy = actor_critic(env=env)
    task = full_control(policy)
    task.test(env=env,name='Tracking')
