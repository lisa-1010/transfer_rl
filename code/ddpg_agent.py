


class DdpgAgent(object):
    def __init__(self):
        self.actor_net = ActorNet()
        self.critic_net = CriticNet()
        self.replay_buffer = ReplayBuffer(buffer_size=1000000)


    def perceive_and_train(self, s, a, r, s_p):
        # puts new info into buffer
        # trains on buffer


    def choose_action(self, s):
        a = self.actor_net.get_action(s)


    def _train(self):
        # implement DDPG Algorithm
        # go through replay buffer
        # for the batch, computer loss for the critic
        # use critic q gradient, pass it into actor,
        # update target networks (both actor and critic)

