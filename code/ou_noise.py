import numpy as np 
import numpy.random as nr

class OU_Noise(object):
	def __init__(self,  action_dim, mu=0, theta=0.15, std=0.3):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.std = std
		self.state = np.ones(action_dim)*self.mu

	def start(self):
		self.state = np.ones(self.action_dim)*self.mu

	def noise(self):
		x = self.state
		delta_x = self.theta*(self.mu - x)+ self.std*nr.randn(self.action_dim)
		self.state = x + delta_x
		return self.state

