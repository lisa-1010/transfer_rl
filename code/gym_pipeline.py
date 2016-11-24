import gym
import tensorflow as tf
from ddpg_agent import *


class Pipeline(object):
    def __init__(self, env_name='InvertedPendulum-v1'):
        self.env = gym.make(env_name)
        # get action + observation space, pass into agent
        self.agent = DdpgAgent()

    def _run_episode(self):
        self.env.reset()

        for timestep in xrange(self.env.spec.timestep_limit):
            action = self.agent.get_action(noise=None)
            # execute in environment
            # perceive new information and train


    def run_test(self):
        # run current agent model on environment,
        # evaluate average reward, create video


def run_training_pipeline(num_episodes=100000):
    pipeline = Pipeline(env_name='InvertedPendulum-v1')

    for episode in xrange(num_episodes):
        pipeline._run_episode()


        # Every 100 episodes, run test and print average reward
        if episode % 100 == 0:
            pipeline.run_test()




