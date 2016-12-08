# creates different fc neural net architectures, returns the tensor with placeholders

import tensorflow as tf
import numpy as np

BATCH_NORM_EPSILON = 1e-5

def init_variable_with_fan_in(shape, fan_in=None):
    if fan_in == None:
        fan_in = shape[0]

    low = - 1/np.sqrt(fan_in)
    high = -low
    return tf.random_uniform(shape, low, high)


def create_q_critic_net(state_dim, action_dim, layer_1_dim, layer_2_dim, merge_mode='elemwise_sum', batch_norm=False):
    W1_state = tf.Variable(init_variable_with_fan_in([state_dim, layer_1_dim]), name='critic_W1_state')
    b1_state = tf.Variable(init_variable_with_fan_in([layer_1_dim], state_dim), name='critic_b1_state')

    W2_state = tf.Variable(init_variable_with_fan_in([layer_1_dim, layer_2_dim]), name='critic_W2_state')

    W1_action = tf.Variable(init_variable_with_fan_in([action_dim, layer_1_dim]), name='critic_W1_action')
    b1_action = tf.Variable(init_variable_with_fan_in([layer_1_dim], action_dim), name='critic_b1_action')

    W2_action = tf.Variable(init_variable_with_fan_in([action_dim, layer_2_dim]), name='critic_W2_action') 

    b2 = tf.Variable(init_variable_with_fan_in([layer_2_dim], layer_1_dim),  name='critic_b2')

    output_dim = 1
    W3 = tf.Variable(tf.random_uniform([layer_2_dim, output_dim], -3e-3, 3e-3), name='critic_W3')
    b3 = tf.Variable(tf.random_uniform([output_dim], -3e-3, 3e-3), name='critic_b3')

    net_vars = [W1_state, b1_state, W2_state, W2_action, b2, W3, b3]

    state_input = tf.placeholder("float32", [None, state_dim], name='critic_state_input')
    action_input = tf.placeholder("float32", [None, action_dim], name='critic_action_input')
    is_training = tf.placeholder(tf.bool, [])
    if batch_norm:
        state_input_bn = performBatchNormalization(state_input, 'critic_state_input_layer', is_training, tf.identity)
        h1_state_prebn = tf.matmul(state_input_bn, W1_state) + b1_state
        h1_state = performBatchNormalization(h1_state_prebn, 'critic_state_layer_1',is_training, tf.nn.relu)
    else:
        h1_state = tf.nn.relu(tf.matmul(state_input, W1_state) + b1_state, name='critic_h1_state_layer')
    
    h2_state = tf.matmul(h1_state, W2_state, name='critic_h2_state_layer')
    h2_action = tf.matmul(action_input, W2_action, name='critic_h2_action_layer')
    h2 = None
    if merge_mode == 'elemwise_sum':
        h2 = tf.nn.relu(tf.add(h2_state, h2_action) + b2)
    elif merge_mode == 'elemwise_mul':
        h2 = tf.nn.relu(tf.mul(h2_state, h2_action) + b2)
    elif merge_mode == 'concat':
        h2 = tf.nn.relu(tf.concat(1, [h2_state, h2_action]) + b2)

    q_value_pred = tf.identity(tf.matmul(h2, W3) + b3) # to remove dimensions of size 1.

    return state_input, is_training, action_input, q_value_pred, net_vars


def create_target_q_critic_net(state_dim, action_dim, net_vars, merge_mode='elemwise_sum', batch_norm=False):
    # Create an ExponentialMovingAverage object
    ema = tf.train.ExponentialMovingAverage(decay=0.999)

    target_update = ema.apply(net_vars)
    target_net_vars = [ema.average(x) for x in net_vars] # moving averages of net_vars in q_critic_net

    W1_state, b1_state,W2_state, W2_action, b2, W3, b3 = tuple(target_net_vars) 

    state_input = tf.placeholder("float", [None, state_dim])
    action_input = tf.placeholder("float", [None, action_dim])
    is_training = tf.placeholder(tf.bool, [])
    if batch_norm:
        state_input_bn = performBatchNormalization(state_input, 'critic_target_state_input_layer', is_training, tf.identity)
        h1_state_prebn = tf.matmul(state_input_bn, W1_state) + b1_state
        h1_state = performBatchNormalization(h1_state_prebn, 'critic_target_state_layer_1',is_training, tf.nn.relu)
    else:
        h1_state = tf.nn.relu(tf.matmul(state_input, W1_state) + b1_state)
    h2_state = tf.matmul(h1_state, W2_state)
    h2_action = tf.matmul(action_input, W2_action)

    h2 = None
    if merge_mode == 'elemwise_sum':
        h2 = tf.nn.relu(tf.add(h2_state, h2_action) + b2)
    elif merge_mode == 'elemwise_mul':
        h2 = tf.nn.relu(tf.mul(h2_state, h2_action) + b2)
    elif merge_mode == 'concat':
        h2 = tf.nn.relu(tf.concat(1, [h2_state, h2_action]) + b2)

    q_value_target = tf.identity(tf.matmul(h2, W3) + b3)  # to remove dimensions of size 1.

    return state_input, is_training, action_input, q_value_target, target_update

def create_actor_net(state_dim, action_dim, layer_1_dim, layer_2_dim, batch_norm=False):
    W1 = tf.Variable(init_variable_with_fan_in([state_dim, layer_1_dim]), name='actor_W1')
    W2 = tf.Variable(init_variable_with_fan_in([layer_1_dim, layer_2_dim]), name='actor_W2')
    W3 = tf.Variable(tf.random_uniform([layer_2_dim, action_dim],  -3e-3, 3e-3), name='actor_W3')
    b1 = tf.Variable(init_variable_with_fan_in([layer_1_dim], state_dim), name='actor_b1')
    b2 = tf.Variable(init_variable_with_fan_in([layer_2_dim], layer_1_dim),  name='actor_b2')
    b3 = tf.Variable(tf.random_uniform([action_dim],  -3e-3, 3e-3), name='actor_b3')
    net_vars = [W1, b1, W2, b2, W3, b3]

    state_input = tf.placeholder("float32", [None, state_dim], name='actor_state_input')
    is_training = tf.placeholder(tf.bool, [])
    if batch_norm:
        input_bn = performBatchNormalization(state_input, 'actor_input_layer', is_training, tf.identity)
        h1 = tf.matmul(input_bn, W1) + b1
        h1_bn = performBatchNormalization(h1, 'actor_layer_1',is_training, tf.nn.relu)
        h2 = tf.matmul(h1_bn, W2) + b2
        h2 = performBatchNormalization(h2, 'actor_layer_2', is_training, tf.nn.relu)
    else:
        h1 = tf.nn.relu(tf.matmul(state_input, W1) + b1, name='actor_h1_layer')
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2,  name='actor_h2_layer')

    actions = tf.nn.tanh(tf.matmul(h2, W3) + b3, name='actor_action_output_layer') # causes actions to be between -1 and 1. Will need to un-nomrmalize when executing
    return state_input, is_training, actions, net_vars


def create_target_actor_net(state_dim, action_dim, net_vars, merge_mode='elemwise_sum', batch_norm=False):
    # Create an ExponentialMovingAverage object
    ema = tf.train.ExponentialMovingAverage(decay=0.999)

    target_update = ema.apply(net_vars)
    target_net_vars = [ema.average(x) for x in net_vars] # moving averages of net_vars in q_critic_net

    W1, b1, W2, b2, W3, b3 = tuple(target_net_vars)
    state_input = tf.placeholder("float", [None, state_dim])
    is_training = tf.placeholder(tf.bool)
    if batch_norm:
        input_bn = performBatchNormalization(state_input, 'actor_target_input_layer', is_training, tf.identity)
        h1 = tf.matmul(input_bn, W1) + b1
        h1_bn = performBatchNormalization(h1, 'actor_target_layer_1',is_training, tf.nn.relu)
        h2 = tf.matmul(h1_bn, W2) + b2
        h2 = performBatchNormalization(h2, 'actor_target_layer_2', is_training, tf.nn.relu)
    else:
        h1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    target_actions = tf.nn.tanh(tf.matmul(h2, W3) + b3) # causes actions to be between -1 and 1. Will need to un-nomrmalize when executing
    return state_input, is_training, target_actions, target_update

def performBatchNormalization(pre_activations, layer_name, is_training, activation):
    training_result = lambda: tf.contrib.layers.batch_norm(pre_activations, activation_fn=activation, \
        epsilon=BATCH_NORM_EPSILON, center=True, scale=True, is_training=True, reuse=None, updates_collections=None, decay=0.9, scope=layer_name)
    testing_result = lambda: tf.contrib.layers.batch_norm(pre_activations, activation_fn=activation, \
        epsilon=BATCH_NORM_EPSILON, center=True, scale=True, is_training=False, reuse=True, updates_collections=None, decay=0.9, scope=layer_name)
    return tf.cond(is_training, training_result, testing_result)

    
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
