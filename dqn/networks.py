import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, n_states, n_actions, activation, lr, loss_func, optim, device):
        super(DQN, self).__init__()
        fc_size_list = [n_states, 64, 64, n_actions]
        self.fc_net = self.create_fc_net(fc_size_list, activation)

        self.loss_func = loss_func
        self.optimizer = optim(self.parameters(), lr=lr)
        self.device = device
        self.to(device)

    def create_fc_net(self, fc_size_list, activation):
        fc_layers = []
        for i in range(len(fc_size_list)-1):
            layer = nn.Linear(fc_size_list[i], fc_size_list[i+1])
            if i == len(fc_size_list) - 2:
                activation = nn.Identity()
            fc_layers += ((layer, activation))
        return nn.Sequential(*fc_layers)

    def forward(self, x):
        q = self.fc_net(x)
        return q

    def sample_discrete_action(self, s, epsilon, n_actions):
        if np.random.uniform(0,1) < epsilon:
            a = np.random.choice(n_actions)
        else:
            a = torch.argmax(self.forward(torch.from_numpy(s).to(self.device))).item()
        return a


class DuelingDQN(nn.Module):
    def __init__(self, n_states, n_actions, activation, lr, loss_func, optim, device):
        super(DuelingDQN, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.act = activation

        self.create_fc_net()

        self.loss_func = loss_func
        self.optimizer = optim(self.parameters(), lr=lr)
        self.device = device
        self.to(device)

    def create_fc_net(self):
        # initial feature layer
        self.l1 = nn.Linear(self.n_states, 64)
        self.l2 = nn.Linear(64, 64)
        # advantage layer
        self.a1 = nn.Linear(64, self.n_actions)
        # state value layer
        self.v1 = nn.Linear(64, 1)
        
    def forward(self, x):
        f1 = self.act(self.l1(x))
        f2 = self.act(self.l2(f1))
        adv = self.act(self.a1(f2))
        val = self.act(self.v1(f2))

        q = val + (adv - torch.mean(adv))
        return q

    def sample_discrete_action(self, s, epsilon, n_actions):
        if np.random.uniform(0,1) < epsilon:
            a = np.random.choice(n_actions)
        else:
            a = torch.argmax(self.forward(torch.from_numpy(s).to(self.device))).item()
        return a
