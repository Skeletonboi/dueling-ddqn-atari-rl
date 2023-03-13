import os, sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from tqdm import tqdm
from replay_buffer import ExperienceReplay
# import envpool

class DQN(nn.Module):
    def __init__(self, fc_size_list, activation, lr, loss_func):
        super(DQN, self).__init__()
        self.fc_net = self.create_fc_net(fc_size_list, activation)

        self.loss_func = loss_func
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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

def main():
    # Set seed
    torch.manual_seed(0)
    # Hyperparameters
    N_EPOCH = 50
    N_STEPS = 200
    LR = 0.001
    LOSS = nn.MSELoss()
    # Initialize env
    env = gym.make('CartPole-v1')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    # Initialize online and target q networks
    online_qnet = DQN(fc_size_list=[n_states, 64, n_actions], activation=nn.ReLU(), lr=LR, loss_func=LOSS)
    target_qnet = DQN(fc_size_list=[n_states, 64, n_actions], activation=nn.ReLU(), lr=LR, loss_func=LOSS)
    buffer = ExperienceReplay(max_buffer_size=50000)

    # Main loop

    for i in tqdm(range(N_EPOCH)):
        # Train one episode
        s, info = env.reset()
        for i in range(N_STEPS):
            # select eps-greedy action
            a = ...
            next_s, rew, done, trunc, info = env.step(a)
            if 




if __name__ == '__main__':
    main()