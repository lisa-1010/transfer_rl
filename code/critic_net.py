import tensorflow as tf
import tflearn
# from network_architectures import *

LAYER_1_DIM = 128
LAYER_2_DIM = 64

class CriticNet(object):

    def __init__(self):
        self.create_critic_with_update_rule()


    def create_critic_with_update_rule(self):
        net = create_3_layer_net(input_dim, LAYER_1_DIM, LAYER_2_DIM)
        pass


    def train_actor(self):
        pass


    def create_target_actor(self):
        pass


    def update_target_actor(self):
        pass


    def save_network(self):
        pass




