# creates different fc neural net architectures, returns the tensor with placeholders

import tensorflow as tf
import tflearn
import numpy as np


def init_variable_with_fan_in(shape, fan_in=None):
    if fan_in == None:
        fan_in = shape[0]

    low = - 1/np.sqrt(fan_in)
    high = -low
    return tf.random_uniform(shape, low, high)


def create_q_critic_net(state_dim, action_dim, layer_1_dim, layer_2_dim, merge_mode='elemwise_sum'):
    W1_state = tf.Variable(init_variable_with_fan_in([state_dim, layer_1_dim]), name='W1_state')
    b1_state = tf.Variable(init_variable_with_fan_in([layer_1_dim], state_dim), name='b1_state')

    W2_state = tf.Variable(init_variable_with_fan_in([layer_1_dim, layer_2_dim]), name='W2_state')

    W1_action = tf.Variable(init_variable_with_fan_in([action_dim, layer_1_dim]), name='W1_action')
    b1_action = tf.Variable(init_variable_with_fan_in([layer_1_dim], action_dim), name='b1_action')

    W2_action = tf.Variable(init_variable_with_fan_in([layer_1_dim, layer_2_dim]), name='W2_action')

    b2 = tf.Variable(init_variable_with_fan_in([layer_2_dim], layer_1_dim),  name='b2')

    output_dim = 1
    W3 = tf.Variable([layer_2_dim, output_dim], -5e-3, 5e-3, name='W3')
    b3 = tf.Variable([output_dim], -5e-3, 5e-3, name='b3')

    net_vars = [W1_state, b1_state, W1_action, b1_action, W2_state, W2_action, b2, W3, b3]
    state_input = tf.placeholder("float", [None, state_dim])
    action_input = tf.placeholder("float", [None, action_dim])

    h1_state = tf.nn.relu(tf.matmul(state_input, W1_state) + b1_state)
    h2_state = tf.matmul(h1_state, W2_state)

    h1_action = tf.nn.relu(tf.matmul(action_input, W1_action) + b1_action)
    h2_action = tf.matmul(h1_action, W2_action)
    h2 = None
    if merge_mode == 'elemwise_sum':
        h2 = tf.nn.relu(tf.add(h2_state, h2_action) + b2)
    elif merge_mode == 'elemwise_mul':
        h2 = tf.nn.relu(tf.mul(h2_state, h2_action) + b2)
    elif merge_mode == 'concat':
        h2 = tf.nn.relu(tf.concat(1, [h2_state, h2_action]) + b2)

    q_value_pred = tf.squeeze(tf.matmul(h2, W3) + b3) # to remove dimensions of size 1.

    return state_input, action_input, q_value_pred, net_vars


def create_target_q_critic_net(state_dim, action_dim, net_vars, merge_mode='elemwise_sum'):
    # Create an ExponentialMovingAverage object
    ema = tf.train.ExponentialMovingAverage(decay=0.999)

    target_update = ema.apply(net_vars)
    target_net_vars = [ema.average(x) for x in net_vars] # moving averages of net_vars in q_critic_net

    W1_state, b1_state, W1_action, b1_action, W2_state, W2_action, b2, W3, b3 = tuple(target_net_vars)
    state_input = tf.placeholder("float", [None, state_dim])
    action_input = tf.placeholder("float", [None, action_dim])

    h1_state = tf.nn.relu(tf.matmul(state_input, W1_state) + b1_state)
    h2_state = tf.matmul(h1_state, W2_state)

    h1_action = tf.nn.relu(tf.matmul(action_input, W1_action) + b1_action)
    h2_action = tf.matmul(h1_action, W2_action)
    h2 = None
    if merge_mode == 'elemwise_sum':
        h2 = tf.nn.relu(tf.add(h2_state, h2_action) + b2)
    elif merge_mode == 'elemwise_mul':
        h2 = tf.nn.relu(tf.mul(h2_state, h2_action) + b2)
    elif merge_mode == 'concat':
        h2 = tf.nn.relu(tf.concat(1, [h2_state, h2_action]) + b2)

    q_value_target = tf.squeeze(tf.matmul(h2, W3) + b3)  # to remove dimensions of size 1.

    return state_input, action_input, q_value_target, target_update

def create_actor_net(state_dim, action_dim, layer_1_dim, layer_2_dim):
    W1 = tf.Variable(init_variable_with_fan_in([state_dim, layer_1_dim]), name='W1')
    b1 = tf.Variable(init_variable_with_fan_in([layer_1_dim], state_dim), name='b1')

    W2 = tf.Variable(init_variable_with_fan_in([layer_1_dim, layer_2_dim]), name='W2')
    b2 = tf.Variable(init_variable_with_fan_in([layer_2_dim], layer_1_dim),  name='b2_actor')

    W3 = tf.Variable([layer_2_dim, action_dim], -5e-3, 5e-3, name='W3_actor')
    b3 = tf.Variable([action_dim], -5e-3, 5e-3, name='b3_actor')

    net_vars = [W1_state, b1_state, W1_action, b1_action, W2_state, W2_action, b2, W3, b3]
    state_input = tf.placeholder("float", [None, state_dim])

    h1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    actions = tf.nn.tanh(tf.matmul(h2, W3) + b3) # causes actions to be between -1 and 1. Will need to un-nomrmalize when executing

    return state_input, actions, net_vars


def create_target_actor_net(state_dim, action_dim, net_vars, merge_mode='elemwise_sum'):
    # Create an ExponentialMovingAverage object
    ema = tf.train.ExponentialMovingAverage(decay=0.999)

    target_update = ema.apply(net_vars)
    target_net_vars = [ema.average(x) for x in net_vars] # moving averages of net_vars in q_critic_net

    W1, b1, W2, b2, W3, b3 = tuple(target_net_vars)
    state_input = tf.placeholder("float", [None, state_dim])

    h1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    target_actions = tf.nn.tanh(tf.matmul(h2, W3) + b3) # causes actions to be between -1 and 1. Will need to un-nomrmalize when executing

    return state_input, target_actions, target_update
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
