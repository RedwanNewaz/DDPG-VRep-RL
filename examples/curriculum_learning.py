from task import MultiQuad
from models import actor_critic as drl_model
from examples import full_control as task
import numpy as np
import tensorflow as tf
import time
from collections import namedtuple
from functools import reduce


np.random.seed(123)

def drl_agent(task):
    '''
    :param task: tracking
    :return: load actor weight and return actor net
    '''
    task['Tracking']
    agent = task.agent.actor
    print('getting actor')
    return agent

if __name__ == '__main__':
    env = MultiQuad(headless=False, curriculum=False) # obtain environment from simulator
    funcs = [drl_model, task, drl_agent]
    policy = reduce(lambda res, func: func(res), funcs, env.quad_1) # same as foldl in haskell

    robot = namedtuple('robot', ['output', 'policy', 'input']) # define the robot class

    # define instances
    robotA = robot(output=env.quad_1._self_observe, policy=policy, input=env.quad_1._set_actions)
    robotB = robot(output=env.quad_2._self_observe, policy=policy, input=env.quad_2._set_actions)
    robots = [robotA, robotB]

    alpha = 0.95
    dt = 0.0


    # synchronize with simulator with respect to base_robot
    base_robot = env.quad_1
    base_robot.reset()
    graph = tf.get_default_graph()

    # start simulation
    while True:
        s = time.time()
        for q in robots:
            s_t = q.output() # get state of the robotrotor
            s_t = s_t.reshape((1, 1, 12))
            with graph.as_default():
                u_t = q.policy.predict_on_batch(s_t).flatten() # predict action given state
                q.input(u_t)

        base_robot.venv.simxSynchronousTrigger() # sync with simulator
        dt = alpha * dt + (1 - alpha) * (time.time() - s) # compute execution time
        print(f'\r avg time taken {dt:.5f}', end='', flush=True) # display execution time





















