import numpy as np
from replay_buffer import *
import tensorflow as tf

TRAIN_BATCH_SIZE = 128 
DISCOUNT = 0.99
NOISE_MEAN = 0
NOISE_STD = 0.01

class DdpgAgent(object):
    def __init__(self, env):
        session = tf.InteractiveSession()
        self.state_norm_params = (env.observation_space.high, env.observation_space.low)
        self.action_norm_params = (env.action_space.high, env.action_space.low)
        self.actor = ActorNet(session, env.observation_space.shape[0],  env.action_space.shape[0])
        self.critic = CriticNet(session,env.observation_space.shape[0], env.action_space.shape[0])
        self.replay_buffer = ReplayBuffer()


    # Note, for reacher task, the observation space ranges from -inf to inf. Also maybe better to compute mean and diff at start
    def _normalize_state(self, state):
        mean = sum(self.state_norm_params)/2
        diff = (self.state_norm_params[0] - self.state_norm_params[1] ) / 2
        return (state - mean)/ diff

    def _normalize_action(self, action):
        mean = sum(self.action_norm_params)/2
        diff = (self.action_norm_params[0] - self.action_norm_params[1] ) / 2
        return (action - mean)/diff

    def _normalize_reward(self, reward):
        # currently does nothing. Consider normalizing reward
        return reward

    def _un_normalize_action(self, actor_action):
        mean = sum(self.action_norm_params)/2
        diff = (self.action_norm_params[0] - self.action_norm_params[1] ) / 2
        return actor_action*diff + mean

    def perceive_and_train(self, s, a, r, s_p):
        s = self._normalize_state(s)
        s_p = self._normalize_state(s_p)
        a = self._normalize_action(a)
        r = self._normalize_reward(r)
        self.replay_buffer.add_observation((s, a, r, s_p))
        if self.buffer.can_replay():
            self._train()


    def get_noisy_action(self, s):
        action = self.get_action(s) + np.random.normal(NOISE_MEAN, NOISE_STD, np.shape(a)) 
        return action

    def get_action(self, s):
        a = self.actor.get_action(s)
        return self._un_normalize_action(a)


    def _train(self):
        # Get minibatch
        states, actions, rewards, next_states = self.replay_buffer.get_minibatch(TRAIN_BATCH_SIZE)
        
        # Compute train targets 
        policy_advantages = self.critic.get_action_gradients((states, actions))
        target_next_actions = self.actor.compute_target_actions(next_states)
        target_q_values = self.critic.compute_target_q_value((next_states, target_next_actions))
        train_targets = rewards + DISCOUNT*target_q_values

        # Train  and update networks 
        self.critic.train((states, actions, train_targets))
        self.actor.train((states, policy_advantages))
        self.critic.update_target()
        self.actor.update_target()
        






