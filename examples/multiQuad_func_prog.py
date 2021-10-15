from task import MultiQuad
from models import actor_critic
from examples import full_control
import numpy as np
import tensorflow as tf
import time
from collections import namedtuple
from functools import reduce

np.random.seed(123)


class multiQuad_func_prog:
    def __init__(self, quadAB, funcs):
        base_quad = reduce(lambda x, y: x if x.sync else y, quadAB)  # check sync flag for the base class representative
        self.__agent = reduce(lambda res, func: func(res), funcs, base_quad.type)  # recursively apply the functions
        base_quad.type.reset()
        self.__graph = tf.get_default_graph()
        self.__base_quad = base_quad
        self.__quads=quadAB

    def __call__(self, quad):
        s_t = quad._self_observe()
        s_t = s_t.reshape((1, 1, 12))
        with self.__graph.as_default():
            u_t = self.__agent.predict_on_batch(s_t).flatten()
            quad._set_actions(u_t)
        return True

    def run(self):
        alpha = 0.95
        dt = 0.0
        while True:
            s = time.time()
            U = [self(quad_.type) for quad_ in self.__quads]
            self.__base_quad.type.venv.simxSynchronousTrigger()
            dt = alpha * dt + (1 - alpha) * (time.time() - s)
            print(f'\r avg time taken {dt:.5f}', end='', flush=True)

    def quit(self):
        print('quitting vrep')
        self.__base_quad.type.venv.end()




if __name__ == '__main__':
    env = MultiQuad(headless=False)
    quads = namedtuple('quad', ['name', 'type', 'sync'])
    quadAB = [quads(name='A', type=env.quad_1, sync=True),
              quads(name='B', type=env.quad_2, sync=False)]


    def get_actor(task):
        '''
        :param task: tracking
        :return: load actor weight and return actor net
        '''
        task['Tracking']
        agent = task.agent.actor
        print('getting actor')
        return agent


    funcs = [actor_critic, full_control, get_actor]
    mq = multiQuad_func_prog(quadAB=quadAB, funcs=funcs)
    try:
        mq.run()
    except:
        mq.quit()












