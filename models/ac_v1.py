
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.initializers import random_uniform



class actor_critic(object):
    def __init__(self, env,name='u_t'):
        self.nb_actions = env.action_space.shape[0]
        self.actor = self.__actor(env,name)
        self.critic = self.__critic(env)



    def __actor(self,env,name='u_t'):
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape,name=name+'/input'))
        actor.add(Dense(64))
        actor.add(Activation('relu'))
        actor.add(Dense(64))
        actor.add(Activation('relu'))
        actor.add(Dense(self.nb_actions, kernel_initializer=random_uniform(maxval=3e-3, minval=-3e-3)))
        actor.add(Activation('tanh',name=name+'/output'))
        return  actor


    def __critic(self,env):
        action_input = Input(shape=(self.nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dense(1, kernel_initializer=random_uniform(maxval=3e-3, minval=-3e-3))(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        self.action_input=action_input
        return critic

