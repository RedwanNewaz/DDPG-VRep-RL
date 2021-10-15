import vrepper
from vrepper.core import vrepper

import os,time
import numpy as np

import gym
from gym import spaces


class Tracking(gym.Env):
    def __init__(self,headless=False):
        self.headless =headless
        self.venv = venv = vrepper(headless=headless)
        venv.start()
        __path = os.path.join(os.getcwd(),'vrepper/scene/quad_track.ttt')
        venv.load_scene(__path)

        self.quadHandle = venv.get_object_by_name('quadcopter')
        self.targetHandle = venv.get_object_by_name('goal')
        # print('Quadricopter simulation started')

        self.__run_time = 1500 #run time was 1000 before
        self.__count =0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,),dtype=np.float32)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(12,),dtype=np.float32)  # TODO: increase the state
        self.initialize = False
        self.initial_orientaiton= self.quadHandle.get_orientation()
        self.__log = {'progress': 0}



    def _self_observe(self):
        # observe then assign
        self.x_p = self.quadHandle.get_position()
        self.x_q = self.quadHandle.get_orientation()
        g_p = self.targetHandle.get_position()
        g_q = self.initial_orientaiton
        x_v, x_w = self.quadHandle.get_velocity()
        e_p = np.array(g_p) - np.array(self.x_p)
        e_q = np.array(g_q) - np.array(self.x_q)
        e_p_dot = -np.array(x_v)
        e_q_dot = -np.array(x_w)
        self.observation = np.hstack((e_p, e_q, e_p_dot, e_q_dot)).astype('float32')
        return self.observation

    def _set_actions(self,u_t):

        self.quadHandle.set_string_signal(list(map(self.mapping,u_t)), 'rotor')




    @staticmethod
    def mapping(X):
        '''
        :param X: neural nets output
        :return: the range is extended
        '''
        X_std = (X + 1) / (1 + 1)
        X_std = np.clip(X_std,0.,1.)
        min_v=1220
        max_v=1350
        X_scaled = X_std * (max_v - min_v) + min_v
        # print( ' u:', X_std)
        if X_std ==0.0:X_scaled -=50 #only for testing
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
                self.venv.start_blocking_simulation()
                self.venv.make_simulation_synchronous(True)
                self._self_observe()
                _restart =  False
            except:
                print('reset error ')
                self._destroy()
                self.__init__(headless=self.headless)


        return self.observation

    def _destroy(self):
        self.venv.stop_blocking_simulation()
        self.venv.end()

    def get_reward(self, s_t):
        '''
         cube size = [ 10x10x10]m^3
         robot_boundary = [(-5,5)x(-5,5)x(0,10)]
         '''
        _abs=lambda x: list(map(abs,x))
        # find robot position in the cube
        _position = np.array(_abs(self.x_p))
        _position[2]-=5
        _in_cube =  (_position<5.0).all()

        # find robot doesn't flipped
        _orientation = np.array(_abs(self.x_q))
        _not_flipped = (_orientation<3.0).all()
        _flipped = not _not_flipped


        # tracking error
        _Qerr = [s_t[0],s_t[1],s_t[2]] #only position
        _Qerr = np.array(_abs(_Qerr))

        self.__log['progress']=  -10*np.linalg.norm(_Qerr)

        # print( 'quad Status ', _in_cube, _flipped, _orientation, _position)
        if not _in_cube or _flipped:
            self.initialize = True

            r_t = -500
        elif (_Qerr<0.1).all(): # target achieved
            r_t = 100
        else:
            r_t = self.__log['progress']


        return r_t



