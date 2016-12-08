import gym
import tensorflow as tf
from ddpg_agent import *
import matplotlib.pyplot as plt

NUM_TEST_TRIALS = 100
BASELINE_PASS = 950.0

class Pipeline(object):
    def __init__(self, env_name='InvertedPendulum-v1'):
        self.env = gym.make(env_name)
        # get action + observation space, pass into agent
        self.agent = DdpgAgent(self.env)
        self.test_performances = []
        videoCallable = lambda x: (x % 1000) == 0
        self.env.monitor.start('../experiments/baseline+bn' + env_name, video_callable=videoCallable)

    def run_episode(self):
        state = self.env.reset()
        for timestep in xrange(self.env.spec.timestep_limit):
            action = self.agent.get_noisy_action(state)
            next_state, reward, done , info = self.env.step(action)
            self.agent.perceive_and_train(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

    def run_test(self, episode_num):
        # run current agent model on environment,
        # evaluate average reward, create video
        total_reward = 0.0
        for episode in xrange(NUM_TEST_TRIALS):
            state = self.env.reset()
            for step in xrange(self.env.spec.timestep_limit):
                # self.env.render()
                action = self.agent.get_action(state)
                next_state, reward, done , info = self.env.step(action)
                state = next_state
                total_reward += reward
                if done:
                    break
        avg_reward = total_reward / NUM_TEST_TRIALS
        self.test_performances.append(avg_reward)
        print 'Episode: {} Average Reward Per Episode : {} '.format(episode_num,avg_reward)
        return avg_reward

    def run(self, num_episodes):
        episode_passed = 0
        for episode in xrange(num_episodes):
            self.run_episode()

            # Every 100 episodes, run test and print average reward
            if episode % 100 == 0:
                avg_reward = self.run_test(episode)
                if (avg_reward > BASELINE_PASS) and not episode_passed: 
                    episode_passed = episode
        episode_passed = episode_passed if episode_passed else num_episodes
        self.env.monitor.close()
        return episode_passed

    def plot_results(self, title):
        plt.xlabel("Episode")
        plt.ylabel("Average Test Reward")
        plt.plot(self.test_performances)
        plt.savefig(title)


if __name__ == "__main__":
    pipeline = Pipeline()
    episode_passed = pipeline.run(num_episodes=20000)
    title = 'BaselineModel+BN' + str(episode_passed)
    pipeline.plot_results(title)


