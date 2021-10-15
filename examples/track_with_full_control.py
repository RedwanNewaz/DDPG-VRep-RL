from examples import vertical_hover
from task import Tracking
from models import actor_critic
import numpy as np

np.random.seed(123)
class full_control(vertical_hover):
    print('full control class initialized')


if __name__ == '__main__':
    env = Tracking(headless=False)
    env.seed(123)
    policy = actor_critic(env=env)
    task = full_control(policy)
    # task.train(env=env,nb_epochs=10,name='Tracking', load_weights=True)
    task.test(env=env,name='Tracking')
