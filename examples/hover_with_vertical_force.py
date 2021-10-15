import pandas as pd
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from task import Hover
from models import actor_critic
import numpy as np

np.random.seed(123)


class vertical_hover(object):

    def __init__(self, policy, batch_size=128, warm_up =120, max_episode=10000):
        self.actor = policy.actor
        self.critic = policy.critic
        self.action_input = policy.action_input
        self.nb_actions = policy.nb_actions
        self.batch_size = batch_size
        self.warm_up = warm_up
        self.max_episode=max_episode
        self.memory = SequentialMemory(limit=100000, window_length=1)
        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=-0.15, mu=0, sigma=0.6)

    @property
    def agent(self):
        __agent= DDPGAgent(nb_actions=self.nb_actions, actor=self.actor, critic=self.critic, critic_action_input=self.action_input,
                         batch_size=self.batch_size,memory=self.memory,
                        nb_steps_warmup_critic=self.warm_up, nb_steps_warmup_actor=self.warm_up,
                        random_process=self.random_process, gamma=.99, target_model_update=0.005)
        __agent.compile([Adam(lr=0.001), Adam(lr=0.01)], metrics=['mse'])
        return __agent

    def __getitem__(self, name):
        print('loading weights', '#'*20, name)
        self.actor.load_weights('result/ddpg_{}_weights_actor.h5f'.format(name))
        self.critic.load_weights('result/ddpg_{}_weights_critic.h5f'.format(name))

    def train(self, env, nb_epochs, name='Hover', load_weights=False):
        if load_weights:self[name]
        __agent = self.agent
        try:
            history=__agent.fit(env, nb_steps=50000*nb_epochs, visualize=False, verbose=1, nb_max_episode_steps=self.max_episode)
        except:
            print('training stops prematurely')

        __agent.save_weights('result/ddpg_{}_weights.h5f'.format(name), overwrite=True)
        dtf=pd.DataFrame(history.history)
        dtf.to_csv('result/log_{}_with_ddpg.csv'.format(name))

    def test(self,env,name):
        self[name]
        self.agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=self.max_episode)



if __name__ == '__main__':
    env = Hover(headless=False)
    env.seed(123)
    policy = actor_critic(env=env)
    task = vertical_hover(policy)
    task.train(env=env,nb_epochs=1)