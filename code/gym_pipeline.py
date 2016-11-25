import gym
import tensorflow as tf
from ddpg_agent import *

NUM_TEST_TRIALS = 100

class Pipeline(object):
    def __init__(self, env_name='InvertedPendulum-v1'):
        self.env = gym.make(env_name)
        # get action + observation space, pass into agent
        self.agent = DdpgAgent(self.env)

    def run_episode(self):
        state = self.env.reset()
        for timestep in xrange(self.env.spec.timestep_limit):
            action = self.agent.get_noisy_action(state)
            next_state, reward, done , info = self.env.step(action)
            self.agent.perceive_and_train(state, action, reward, next_state)
            state = next_state
            if done:
                break

    def run_test(self):
        # run current agent model on environment,
        # evaluate average reward, create video
        total_reward = 0.0
        for episode in xrange(NUM_TEST_TRIALS):
            state = self.env.reset()
            for step in xrange(self.env.spec.timestep_limit):
                action = self.agent.get_action(state)
                next_state, reward, done , info = self.env.step(action)
                state = next_state
                total_reward += reward
                if done:
                    break
        avg_reward = total_reward / NUM_TEST_TRIALS
        print 'Average Reward Per Episode : ', avg_reward


def run_training_pipeline(num_episodes=100000):
    pipeline = Pipeline(env_name='InvertedPendulum-v1')

    for episode in xrange(num_episodes):
        pipeline.run_episode()

        # Every 100 episodes, run test and print average reward
        if episode % 10 == 0:
            pipeline.run_test()



if __name__ == "__main__":
    run_training_pipeline()


