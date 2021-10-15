#!/home/redwan/anaconda3/envs/ddpg@vrep-rl/bin/python
from task import MultiQuad
from models import actor_critic
from examples import full_control
import numpy as np
import tensorflow as tf
import time
import threading
np.random.seed(123)



class multiQauds:
    alpha=0.95
    dt=0
    def __init__(self,task):
        self.__graph = tf.get_default_graph()
        self.__actor = task.agent.actor
        self.__lock = threading.Lock()

    def __interaction(self, quad):
        try:
            s_t = quad._self_observe()
            s_t = s_t.reshape((1, 1, 12))  # state size is 12
            with self.__graph.as_default():
                u_t = self.__actor.predict_on_batch(s_t).flatten()
                quad._set_actions(u_t)  # num of actions is 4
            return True
        except:
            print('no interaction ')
            return False

    def __interaction2(self, quad):
        try:
            self.__lock.acquire()
            s_t = quad._self_observe()
            s_t = s_t.reshape((1, 1, 12)) # state size is 12
            with self.__graph.as_default():
                u_t = self.__actor.predict_on_batch(s_t).flatten()
                quad._set_actions(u_t) # num of actions is 4
            self.__lock.release()
            return
        except:
            print('no interaction ')
            return


    def __getitem__(self, item):
        result_info = [threading.Event(), None]
        def runit():
            result_info[1] = self.__interaction(item)
            result_info[0].set()

        threading.Thread(target=runit).start()
        return result_info

    def __gather_results__(self,result_infos):
        results = []
        for x in result_infos:
            x[0].wait()
            results.append(x[1])
        return results

    def run2(self,args):
        print('version 2','#'*30)
        while True:
            s = time.time()
            threads = []
            # # Create new threads
            for q in args:
                t = threading.Thread(target=self.__interaction2, args=(q,))
                t.start()
                threads.append(t)
            # # Wait for all threads to complete
            for t in threads:
                t.join()

            self.dt = self.alpha * self.dt + (1 - self.alpha) * (time.time() - s)
            # print(f'\r avg time taken {self.dt:.5f} | result: {np.prod(results)}', flush=True, end='')
            print(f'\r avg time taken {self.dt:.5f}', flush=True, end='')

            args[0].venv.simxSynchronousTrigger()

    def __call__(self, *args, **kwargs):
        while True:
            s = time.time()
            results = self.__gather_results__((self[q] for q in args))
            self.dt = self.alpha * self.dt + (1 - self.alpha) * (time.time() - s)
            print(f'\r avg time taken {self.dt:.5f} | result: {np.prod(results)}', flush=True, end='')
            args[0].venv.simxSynchronousTrigger()


if __name__ == '__main__':
    env = MultiQuad(headless=False)
    quad_1 = env.quad_1
    quad_2 = env.quad_2

    policy = actor_critic(env=quad_1) # any quads will serve the purpose
    task = full_control(policy)
    task['Tracking'] # loading weights (only actor weight is enough for this purpose)
    quad_1.reset() # staring simulation
    mq=multiQauds(task)
    # mq.run2([quad_1,quad_2])
    mq(quad_1, quad_2)










