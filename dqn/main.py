import os, sys
sys.path.append(os.getcwd())
sys.path.append('../utils')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from tqdm import tqdm
from replay_buffers import ExperienceReplay
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
    UPDATE_STEPS = 10

    N_ENVS = 16
    BATCH_SIZE = 16
    BUFFER_SIZE = 50000

    GAMMA = 0.99
    LR = 0.001
    EPS = 0.9
    EPS_DECAY = 0.99
    # Initialize env
    env = gym.make('CartPole-v1')
    # env = envpool.make('CartPole-v1', env_type='gym', num_envs=N_ENVS)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    # Initialize online and target q-networks and exp. replay buffer
    online_qnet = DQN(fc_size_list=[n_states, 64, n_actions], activation=nn.ReLU(), lr=LR, loss_func=nn.MSELoss())
    target_qnet = DQN(fc_size_list=[n_states, 64, n_actions], activation=nn.ReLU(), lr=LR, loss_func=nn.MSELoss())
    exp_replay = ExperienceReplay(max_buffer_size=BUFFER_SIZE)
    # Main loop
    for i in tqdm(range(N_EPOCH)):
        # Train one episode
        s, info = env.reset()
        for i in range(N_STEPS):
            # eps-greedy action sampling
            if np.random.uniform(0,1) < EPS:
                a = np.random.choice(n_actions)
            else:
                a = torch.argmax(online_qnet(torch.from_numpy(s))).item()
            next_s, rew, done, trunc, info = env.step(a)
            exp_replay.insert(s, next_s, a, rew, done)
            # learn 
            if exp_replay.size() > BATCH_SIZE:
                batch_s, batch_ns, batch_a, batch_r, batch_d = exp_replay.sample_experience(BATCH_SIZE)
                # update target net
                if i % UPDATE_STEPS == 0:
                    target_qnet.load_state_dict(online_qnet.state_dict())
                # with torch.no_grad():
                q_pred = online_qnet.forward(batch_s)
                q_targ_ns = target_qnet.forward(batch_ns)
                max_a_ns = torch.argmax(q_targ_ns, dim=1, keepdim=True)
                q_targ = torch.add(batch_r, GAMMA * (1 - batch_d) * q_targ_ns.gather(dim=1, index=max_a_ns))
                
                loss = online_qnet.loss_func(q_pred.gather(dim=1, index=batch_a), q_targ)
                online_qnet.optimizer.zero_grad()
                loss.backward()
                online_qnet.optimizer.step()
            
        EPS = EPS * EPS_DECAY
        print('NEW EPS:', EPS)





if __name__ == '__main__':
    main()