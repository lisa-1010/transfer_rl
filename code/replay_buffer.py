import random

DEFAULT_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 1000

class ReplayBuffer(object):

	def __init__(self, buffer_size=DEFAULT_BUFFER_SIZE):
		self.buffer = []
		self.buffer_size = buffer_size

	def add_observation(self, s, a, r, s_p):
		if len(self.buffer) >= DEFAULT_BUFFER_SIZE:
			self.buffer.pop(0)
		self.buffer.append((s, a, r, s_p))

	def can_replay(self):
		return len(self.buffer) > REPLAY_START_SIZE 

	def get_minibatch(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		states = []
		actions = []
		rewards = []
		next_states = []
		for obs in batch:
			states.append(obs[0])
			actions.append(obs[1])
			rewards.append(obs[2])
			next_states.append(obs[3])
		return states, actions, np.array(rewards), next_states
