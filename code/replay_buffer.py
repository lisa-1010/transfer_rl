import random
import numpy as np

DEFAULT_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000

class ReplayBuffer(object):

    def __init__(self, buffer_size=DEFAULT_BUFFER_SIZE):
        self.buffer = []
        self.buffer_size = buffer_size

    def add_observation(self, s, a, r, s_p, done):
        if len(self.buffer) >= DEFAULT_BUFFER_SIZE:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, s_p, done))

    def can_replay(self):
        return len(self.buffer) > REPLAY_START_SIZE

    def get_minibatch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        for obs in batch:
            states.append(obs[0])
            actions.append(obs[1])
            rewards.append(obs[2])
            next_states.append(obs[3])
            dones.append(obs[4])

        states = np.array(states, dtype='float32')
        # rigging by conducting batch normalization of states before outputing. 
        actions = np.array(actions, dtype='float32')

        # rigging by conducting batch normalization of rewards before outputing.
        rewards = np.expand_dims(np.array(rewards, dtype='float32'), axis=1)

        # rigging by conducting batch normalization of states before outputing. 
        next_states = np.array(next_states, dtype='float32')
        return states, actions, rewards, next_states, dones
