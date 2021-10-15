import vrepper
from vrepper.core import vrepper

import os,time
import numpy as np

import gym
from gym import spaces
import threading
threadLock = threading.Lock()


class _path_follow(gym.Env):
    def __init__(self, venv,quad,goal,cmdname='rotor', curriculum=False):
        self.venv = venv
        self.__curriculum = curriculum
        self.quadHandle = venv.get_object_by_name(quad)
        self.targetHandle = venv.get_object_by_name(goal)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(12,), dtype=np.float32)  # TODO: increase the state
        self.initial_orientaiton = self.quadHandle.get_orientation()
        self.cmdname = cmdname
        self._self_observe()



    def _self_observe(self):
        # observe then assign
        self.x_p = self.quadHandle.get_position()
        self.x_q = self.quadHandle.get_orientation()
        g_p = self.targetHandle.get_position()
        g_q = self.initial_orientaiton
        x_v, x_w = self.quadHandle.get_velocity()
        g_v, g_w = self.targetHandle.get_velocity()
        e_p = np.array(g_p) - np.array(self.x_p)
        e_q = np.array(g_q) - np.array(self.x_q)
        e_p_dot = np.array(g_v) - np.array(x_v)  # has been modified
        e_q_dot = np.array(g_w) - np.array(x_w)
        self.observation = np.hstack((e_p, e_q, e_p_dot, e_q_dot)).astype('float32')

        return self.observation

    def _set_actions(self,u_t):
        self.quadHandle.set_string_signal(list(map(self.mapping,u_t)), self.cmdname)






    def mapping(self,X):
        '''
        :param X: neural nets output
        :return: the range is extended
        '''
        X_std = (X + 1) / (1 + 1)
        X_std = np.clip(X_std,0.,1.)
        if self.__curriculum:
            min_v = 1170  # default 1220
            max_v = 1400  # default 1350
        else:
            min_v=1220
            max_v=1500
        X_scaled = X_std * (max_v - min_v) + min_v
        return X_scaled



    def reset(self):
        # print(' init:{} | prog:{}'.format(self.initialize,self.__log['progress']))

        print(self.cmdname)
        # if self.cmdname is not 'rotor': return self.observation
        _restart = True
        while _restart:
            try:
                # threadLock.acquire()
                self.venv.stop_simulation()
                self.venv.start_blocking_simulation()
                self.venv.make_simulation_synchronous(True)
                self._self_observe()
                _restart =  False
                # threadLock.release()
            except:
                print('reset error ', self.cmdname)




class MultiQuad(_path_follow):
    def __init__(self, headless=False, curriculum=False):
        self.headless =headless
        self.venv = venv = vrepper(headless=headless)
        venv.start()
        __path = os.path.join(os.getcwd(), 'vrepper/scene/multi_quad_path_follow.ttt')
        venv.load_scene(__path)
        self.quad_1 = _path_follow(venv=venv, quad='quadcopter', goal='goal', curriculum=curriculum)
        self.quad_2 = _path_follow(venv=venv, quad='quadrotor', goal='goal0', cmdname='rotor_2', curriculum=curriculum)



