from task import MultiQuad
from models import actor_critic
from examples import full_control
import numpy as np
import threading
import tensorflow as tf
np.random.seed(123)


class myThread (threading.Thread):
   def __init__(self,  name, env):
      threading.Thread.__init__(self)
      self.name = name
      self.env = env



   def run(self):
       while True:
           s_t = self.env._self_observe()
           s_t = s_t.reshape((1, 1, 12))
           u_t = prediction(s_t)
           self.env._set_actions(u_t)
           self.env.simxSynchronousTrigger()



def prediction(s_t):
    threadLock.acquire()
    with graph.as_default():
        u_t = agent.predict_on_batch(s_t).flatten()
    threadLock.release()
    return u_t

if __name__ == '__main__':
    env = MultiQuad(headless=False)
    quad_1 = env.quad_1
    quad_2 = env.quad_2

    policy = actor_critic(env=quad_2, name='universal')
    task = full_control(policy)
    graph = tf.get_default_graph()
    task['Tracking']
    agent = task.agent.actor

    threadLock = threading.Lock()
    threads = []

    # Create new threads
    thread1 = myThread("Quad1", quad_1)
    thread2 = myThread("Quad2", quad_2)
    #
    # Start new Threads
    thread1.start()
    thread2.start()

    quad_1.reset()

    threads.append(thread1)
    threads.append(thread2)
    for t in threads:
        t.join()