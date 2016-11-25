import tensorflow as tf
from network_architectures import *

LAYER_1_DIM = 400
LAYER_2_DIM = 300


LEARNING_RATE = 0.0001

class ActorNet(object):

    def __init__(self, sess, state_dim, action_dim):

        self.sess = sess # Tensorflow Session

        self.state_input, self.action_output, self.net_vars = \
            create_actor_net(state_dim, action_dim, LAYER_1_DIM, LAYER_2_DIM)

        self.target_state_input, self.target_action_output, self.target_update = \
            create_target_actor_net(state_dim, action_dim, self.net_vars)

        self.action_q_gradients = tf.placeholder("float", [None, action_dim]) 
        self.param_gradients = tf.gradients(self.action_output, self.net_vars, -self.action_q_gradients) # TODO: figure out if we need a negative before action_q_gradients
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).apply_gradients(zip(self.param_gradients, self.net_vars))

        self.train_step = 0

        self.sess.run(tf.initialize_all_variables())
        self.update_target()


    def train(self, input_batch):
        state_batch, action_q_gradients = input_batch
        self.train_step += 1
        self.sess.run(self.optimizer, feed_dict={self.state_input:state_batch, self.action_q_gradients:action_q_gradients})


    def update_target(self):
        self.sess.run(self.target_update)


    def get_action(self, input_states):
        actions = self.sess.run(self.action_output, feed_dict={self.state_input:input_states})
        return actions


    def compute_target_actions(self, input_states):
        target_actions  = self.sess.run(self.target_action_output, feed_dict={self.target_state_input:input_states})
        return target_actions


    def save_network(self):
        pass
