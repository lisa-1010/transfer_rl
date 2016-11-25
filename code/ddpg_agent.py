import numpy as np
from replay_buffer import *
import tensorflow as tf

from critic_net import *
from actor_net import *

TRAIN_BATCH_SIZE = 10 
DISCOUNT = 0.99
NOISE_MEAN = 0
NOISE_STD = 0.01

class DdpgAgent(object):
    def __init__(self, env):
        session = tf.InteractiveSession()
        self.state_norm_params = (env.observation_space.high, env.observation_space.low)
        self.action_norm_params = (env.action_space.high, env.action_space.low)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = ActorNet(session, state_dim, action_dim)
        self.critic = CriticNet(session, state_dim, action_dim)
        self.replay_buffer = ReplayBuffer()


    # Note, for reacher task, the observation space ranges from -inf to inf. Also maybe better to compute mean and diff at start
    def _normalize_state(self, state):
        mean = (self.state_norm_params[0] + self.state_norm_params[1] )/2.0
        diff = (self.state_norm_params[0] - self.state_norm_params[1] ) / 2.0
        return (state - mean)/ diff

    def _normalize_action(self, action):
        mean = (self.action_norm_params[0] + self.action_norm_params[1] )/2.0
        diff = (self.action_norm_params[0] - self.action_norm_params[1] ) / 2.0
        return (action - mean)/diff

    def _normalize_reward(self, reward):
        # currently does nothing. Consider normalizing reward
        return reward

    def _un_normalize_action(self, actor_action):
        mean = sum(self.action_norm_params)/2
        diff = (self.action_norm_params[0] - self.action_norm_params[1] ) / 2.0
        return actor_action*diff + mean

    def perceive_and_train(self, s, a, r, s_p):
        # Todo : Figure out how to normalize when the range of values is not given
        a = self._normalize_action(a)
        self.replay_buffer.add_observation(s, a, r, s_p)
        if self.replay_buffer.can_replay():
            self._train()


    def get_noisy_action(self, s):
        a = self.get_action(s)
        a += np.random.normal(NOISE_MEAN, NOISE_STD, np.shape(a))
        return a

    def get_action(self, s):
        a = self.actor.get_action([s])
        return self._un_normalize_action(a)[0]


    def _train(self):
        # Get minibatch
        states, actions, rewards, next_states = self.replay_buffer.get_minibatch(TRAIN_BATCH_SIZE)
        
        # Compute train targets
        target_next_actions = self.actor.compute_target_actions(next_states)
        policy_advantages = self.critic.get_action_gradients((states, actions))
        target_q_values = self.critic.compute_target_q_value((next_states, target_next_actions))
        train_targets = rewards + DISCOUNT*target_q_values

        # Train  and update networks 
        self.critic.train((states, actions, train_targets))
        self.actor.train((states, policy_advantages))
        self.critic.update_target()
        self.actor.update_target()
        






