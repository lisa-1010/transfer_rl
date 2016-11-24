# creates different fc neural net architectures, returns the tensor with placeholders

import tensorflow as tf
import tflearn
import numpy as np

# def create_state_action_three_layer_net(state_dim, action_dim, layer_1_dim, layer_2_dim, output_dim, mode='elemwise_sum'):
#     # with tflearn
#     state_input_layer = tflearn.input_data(shape=[None, state_dim])
#     action_input_layer = tflearn.input_data(shape=[None, action_dim])
#
#     state_net = tflearn.fully_connected(state_input_layer, layer_1_dim)
#     state_net = tflearn.fully_connected(state_net, layer_2_dim)
#     action_net = tflearn.fully_connected(action_input_layer, layer_2_dim)
#
#     net = tflearn.merge([state_net, action_net], mode=mode)
#
#     net = tflearn.fully_connected(net, output_dim)
#     return net


def init_variable_with_fan_in(shape, fan_in=None):
    if fan_in == None:
        fan_in = shape[0]

    low = - 1/np.sqrt(fan_in)
    high = -low
    return tf.random_uniform(shape, low, high)


def create_q_critic_net(state_dim, action_dim, layer_1_dim, layer_2_dim, mode='elemwise_sum'):
    W1_state = tf.Variable(init_variable_with_fan_in([state_dim, layer_1_dim]))
    b1_state = tf.Variable(init_variable_with_fan_in([layer_1_dim], state_dim))

    W2_state = tf.Variable(init_variable_with_fan_in([layer_1_dim, layer_2_dim]))

    W1_action = tf.Variable(init_variable_with_fan_in([action_dim, layer_1_dim]))
    b1_action = tf.Variable(init_variable_with_fan_in([layer_1_dim], action_dim))

    W2_action = tf.Variable(init_variable_with_fan_in([layer_1_dim, layer_2_dim]))

    b2 = tf.Variable(init_variable_with_fan_in([layer_2_dim], layer_1_dim))

    output_dim = 1
    W3 = tf.Variable([layer_2_dim, output_dim], -5e-3, 5e-3)
    b3 = tf.Variable([output_dim], -5e-3, 5e-3)

    variables = [W1_state, b1_state, W1_action, b1_action, W2_state, W2_action, b2, W3, b3]
    state_input_layer = tf.placeholder("float", [None, state_dim])
    action_input_layer = tf.placeholder("float", [None, action_dim])

    h1_state = tf.nn.relu(tf.matmul(state_input_layer, W1_state) + b1_state)
    h2_state = tf.matmul(h1_state, W2_state)

    h1_action = tf.nn.relu(tf.matmul(action_input_layer, W1_action) + b1_action)
    h2_action = tf.matmul(h1_action, W2_action)
    h2 = tf.nn.relu(tf.add(h2_state, h2_action) + b2)

    q_value = tf.squeeze(tf.matmul(h2, W3) + b3) # to remove dimensions of size 1.

    return q_value, variables

