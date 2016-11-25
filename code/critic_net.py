import tensorflow as tf
import tflearn
from network_architectures import *
import numpy as np

LAYER_1_DIM = 128
LAYER_2_DIM = 64

L2_FACTOR = 0.01
MERGE_MODE = 'elemwise_sum'

class CriticNet(object):

    def __init__(self, sess, state_dim, action_dim):

        self.sess = sess # Tensorflow Session

        self.state_input, self.action_input, self.q_value_output, self.net_vars = \
            create_q_critic_net(state_dim, action_dim, LAYER_1_DIM, LAYER_2_DIM, merge_mode=MERGE_MODE)

        self.target_state_input, self.target_action_input, self.target_q_value_output, self.target_update = \
            create_target_q_critic_net(state_dim, action_dim, self.net_vars, merge_mode=MERGE_MODE)

        self.target_q_value_input = tf.placeholder("float", [None, 1])

        weight_decay = tf.add_n([L2_FACTOR * tf.nn.l2_loss(var) for var in self.net_vars])
        self.loss = tf.reduce_mean(tf.square(self.target_q_value_input - self.q_value_output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self.action_gradients = tf.gradients(self.q_value_output, self.action_input)
        self.state_gradients = tf.gradients(self.q_value_output, self.state_input)

        self.train_step = 0

        self.sess.run(tf.initialize_all_variables())
        self.update_target()


    def train(self, input_batch):
        state_batch, action_batch, target_batch = input_batch
        self.train_step += 1
        self.sess.run(self.optimizer, feed_dict={self.state_input: state_batch, self.action_input: action_batch,
                                                 self.target_q_value_input: target_batch})


    def update_target(self):
        self.sess.run(self.target_update)


    def compute_q_value(self, input_batch):
        # input_batch is a tuple of state_batch and action_batch
        state_batch, action_batch = input_batch
        q_value = self.sess.run(self.q_value,
                      feed_dict={self.state_input: state_batch, self.action_input: action_batch})
        return q_value


    def compute_target_q_value(self, input_batch):
        state_batch, action_batch = input_batch
        target_q_value = self.sess.run(self.target_q_value_output,
                      feed_dict={self.target_state_input: state_batch, self.target_action_input: action_batch})
        return target_q_value


    def get_action_gradients(self, input_batch):
        state_batch, action_batch = input_batch

        action_grads = self.sess.run(self.action_gradients,
                      feed_dict={self.state_input: state_batch, self.action_input: action_batch})[0]
        return action_grads


    def save_network(self):
        pass




