import vrepper
from vrepper.core import vrepper

import os,time
import numpy as np

import gym
from gym import spaces


class Hover(gym.Env):
    def __init__(self,headless=False):
        self.headless =headless
        self.venv = venv = vrepper(headless=headless)
        venv.start()
        __path = os.path.join(os.getcwd(),'vrepper/scene/quad_hover.ttt')
        venv.load_scene(__path)

        self.quadHandle = venv.get_object_by_name('quadcopter')
        self.targetHandle = venv.get_object_by_name('goal')
        # print('Quadricopter simulation started')

        self.__run_time = 1000
        self.__count =0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,),dtype=np.float32)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(12,),dtype=np.float32)  # TODO: increase the state
        self.initialize = False
        self.initial_orientaiton= self.quadHandle.get_orientation()
        self.__log = {'progress': 0}



    def _self_observe(self):
        # observe then assign
        x_p = self.quadHandle.get_position()
        x_q = self.quadHandle.get_orientation()
        g_p = self.targetHandle.get_position()
        g_q = self.initial_orientaiton
        x_v, x_w = self.quadHandle.get_velocity()
        e_p = np.array(g_p) - np.array(x_p)
        e_q = np.array(g_q) - np.array(x_q)
        e_p_dot = -np.array(x_v)
        e_q_dot = -np.array(x_w)
        self.observation = np.hstack((e_p, e_q, e_p_dot, e_q_dot)).astype('float32')
        return self.observation

    def _set_actions(self,u_t):

        self.quadHandle.set_string_signal(self.mapping(u_t), 'rotor')




    @staticmethod
    def mapping(X):
        X_std = (X + 1) / (1 + 1)
        X_std = np.clip(X_std,0.,1.)
        min_v=1200
        max_v=1300
        X_scaled = X_std * (max_v - min_v) + min_v
        # print( ' u:', X_scaled)
        return X_scaled


    def step(self,u_t):
        self.__count += 1
        # step
        try:
            self._set_actions(u_t)
            self.venv.simxSynchronousTrigger()
            self._self_observe()
        except:
            print('step cannot perform')
        finally:
            # observe again
            s_t=self.observation

            # cost
            r_t = self.get_reward(s_t)
            done = self.initialize
            if self.__count>self.__run_time:
                done = True
                self.initialize = True
                self.__count=0

            if self.initialize:
                self.reset()
                self.initialize=False

        return s_t, r_t, done, self.__log

    def reset(self):
        # print(' init:{} | prog:{}'.format(self.initialize,self.__log['progress']))
        _restart = True
        while _restart:
            try:
                self.venv.stop_simulation()
                # self.venv.simxFinish(-1)
                self.venv.start_blocking_simulation()
                self.venv.make_simulation_synchronous(True)
                self._self_observe()
                _restart =  False
            except:
                print('reset error ')
                self._destroy()
                self.__init__(headless=self.headless)

            # finally:
            #     time.sleep(1)

        return self.observation

    def _destroy(self):
        self.venv.stop_blocking_simulation()
        self.venv.end()

    def get_reward(self, s_t):
        '''
         This function returns reward based on current state and the bounding box
         '''

        z=abs(s_t[2])
        # print(' z:',z)
        __progress = -10*z
        self.__log = {'progress': __progress}
        r_t=__progress

        if z < 0.1:
            r_t = 10

        if z>= 10.0:
            # r_t = -500
            self.initialize = True

        # print('reward: {:.3f}'.format(r_t))

        return r_t



