from task import PathFollow
from models import actor_critic
from examples import full_control
import numpy as np
import tensorflow as tf
import time
from collections import defaultdict
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(123)

def record_state(history,data):
    for key,val in zip(fields,data):
        history[key].append(val)
    return history
def record_action(history,data):
    for key,val in zip(forces,data):
        history[key].append(val)
    return history


def run():
    dt = 0
    alpha = .95
    step = 0
    graph = tf.get_default_graph()
    history = defaultdict(list)
    while step<2600:
        t1 =  time.time()
        s_t = env._self_observe()
        history = record_state(history,s_t)
        s_t = s_t.reshape((1, 1, 12))  # state size is 12
        with graph.as_default():
            u_t = actor.predict_on_batch(s_t).flatten()
            env._set_actions(u_t)
            history = record_action(history, u_t)
        env.venv.simxSynchronousTrigger()
        t2 = time.time()
        dt = alpha*dt + (1-alpha)*(t2-t1)
        step+=1

        print(f'\r execution time: {dt:.3f}| step: {step}', end='',flush=True)
    return history

def get_actor(task):
    '''
    :param task: tracking
    :return: load actor weight and return actor net
    '''
    task['Tracking']
    agent = task.agent.actor
    print('getting actor')
    return agent

if __name__ == '__main__':

    fields = 'x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot'
    fields = fields.split(',')
    forces = ['f1', 'f2', 'f3', 'f4']


    env = PathFollow(headless=False)
    funcs = [actor_critic,full_control,get_actor]
    actor = reduce(lambda res, func: func(res),funcs,env)

    env.reset()
    history=run()
    dtf = pd.DataFrame(history)
    dtf.to_csv('result/inp_out_path_follow.csv')
    print('\n terminating program')
    env._destroy()


    data = pd.read_csv('result/inp_out_path_follow.csv')
    data[70:].plot(x='x', y='y')
    plt.show()




