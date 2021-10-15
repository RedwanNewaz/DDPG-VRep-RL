from task import MultiQuad
from models import actor_critic
from examples import full_control
import numpy as np
import tensorflow as tf
import time
np.random.seed(123)



def prediction(s_t):
    with graph.as_default():
        u_t = agent.predict_on_batch(s_t).flatten()
    return u_t


def get_input(quad):
    s_t = quad._self_observe()
    s_t = s_t.reshape((1, 1, 12))
    u_t = prediction(s_t)
    quad._set_actions(u_t)
    return True

if __name__ == '__main__':
    env = MultiQuad(headless=False)
    quad_1 = env.quad_1
    quad_2 = env.quad_2

    policy = actor_critic(env=quad_2, name='universal')
    task = full_control(policy)
    graph = tf.get_default_graph()
    task['Tracking']
    agent = task.agent.actor

    quad_1.reset()
    quads= [quad_1,quad_2]
    alpha = 0.95
    dt = 0.0
    while True:
        s = time.time()
        U = [get_input(quad_) for quad_ in quads]
        quad_1.venv.simxSynchronousTrigger()
        dt = alpha * dt + (1 - alpha) * (time.time() - s)
        print(f'\r avg time taken {dt:.5f}', end='', flush=True)









