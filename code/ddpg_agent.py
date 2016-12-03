import numpy as np
from replay_buffer import *
import tensorflow as tf

from critic_net import *
from actor_net import *

TRAIN_BATCH_SIZE = 64
DISCOUNT = 0.99
REWARD_NORMALIZATION =  1.0 # Depends on task. Investigate - take (SUCCESS_REWARD/EXPECTED_TRIALS*TIME_PER_TRIAL)

# HIGH PRIORITY
# TODO: Implement Batch Normalization
# TODO: Need to simulate Ornstein-Uhlenbeck process to return noisy action.

# LOW PRIORITY
# TODO: Implement TPRO
# TODO: Using Value Function instead of Q function - Generalized advantage estimation. 

class DdpgAgent(object):
    def __init__(self, env):
        session = tf.InteractiveSession()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = ActorNet(session, self.state_dim, self.action_dim)
        self.critic = CriticNet(session, self.state_dim, self.action_dim)
        self.replay_buffer = ReplayBuffer()
        self._compute_norm_params(env.observation_space.high, env.observation_space.low, env.action_space.high, env.action_space.low)

    def _compute_norm_params(self, state_high, state_low, action_high, action_low):
        self.state_mean = (state_high + state_low ) /2.0
        self.state_diff = (state_high - state_low) / 2.0
        self.action_mean = (action_high + action_low) / 2.0
        self.action_diff = (action_high - action_low) / 2.0


    # Note, for reacher task, the observation space ranges from -inf to inf. Also maybe better to compute mean and diff at start
    def _normalize_state(self, state):
        return (state - self.state_mean)/ self.state_diff

    def _normalize_action(self, action):
        return (action - self.action_mean)/self.action_diff

    def _normalize_reward(self, reward):
        # currently does nothing. Consider normalizing reward
        return REWARD_NORMALIZATION*reward

    def _un_normalize_action(self, actor_action):
        return actor_action*self.action_diff + self.action_mean

    def perceive_and_train(self, s, a, r, s_p, done):
        # TODO : Figure out how to normalize when the range of values is not given
        a = self._normalize_action(a)
        r = self._normalize_reward(r)
        self.replay_buffer.add_observation(s, a, r, s_p, done)
        if self.replay_buffer.can_replay():
            self._train()


    def get_noisy_action(self, s):
        a = self.get_action(s)
        # TODO: Need to simulate Ornstein-Uhlenbeck process to return noisy action.
        return a

    def get_action(self, s):
        a = self.actor.get_action([s])
        return a[0]


    def _train(self):
        # Get minibatch
        states, actions, rewards, next_states, dones = self.replay_buffer.get_minibatch(TRAIN_BATCH_SIZE)
        
        # Compute train targets
        target_next_actions = self.actor.compute_target_actions(next_states)
        target_q_values = self.critic.compute_target_q_value((next_states, target_next_actions))
        train_targets = []
        for index, reward in enumerate(rewards):
            ttarget = reward
            if not dones[index]:
                ttarget += DISCOUNT*target_q_values[index] 
            train_targets.append(ttarget)
        train_targets = np.resize(train_targets, [TRAIN_BATCH_SIZE, 1])
        self.critic.train((states, actions, train_targets))

        # Train  and update networks
        noiseless_actions = self.actor.get_action(states)
        policy_advantages = self.critic.get_action_gradients((states, noiseless_actions))
        self.actor.train((states, policy_advantages))

        self.critic.update_target()
        self.actor.update_target()



