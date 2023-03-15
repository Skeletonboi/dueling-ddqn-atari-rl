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
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device='cpu'
    # Hyperparameters
    N_EPOCH = 5000
    N_STEPS = 200
    UPDATE_STEPS = 4

    N_ENVS = 16
    BATCH_SIZE = 16
    BUFFER_SIZE = 50000

    GAMMA = 0.99
    LR = 0.0001
    INIT_EPS = 0.1
    FIN_EPS = 0.0001
    EXPLORE = 20000
    epsilon = INIT_EPS
    # Initialize env
    env = gym.make('CartPole-v1')
    # env = envpool.make('CartPole-v1', env_type='gym', num_envs=N_ENVS)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    # Initialize online and target q-networks and exp. replay buffer
    online_qnet = DQN(fc_size_list=[n_states, 64, 256, n_actions], activation=nn.ReLU(), lr=LR, loss_func=nn.MSELoss()).to(device)
    target_qnet = DQN(fc_size_list=[n_states, 64, 256, n_actions], activation=nn.ReLU(), lr=LR, loss_func=nn.MSELoss()).to(device)
    target_qnet.load_state_dict(online_qnet.state_dict())
    exp_replay = ExperienceReplay(max_buffer_size=BUFFER_SIZE)
    # Main loop
    for epoch in tqdm(range(N_EPOCH)):
        # Train one episode
        s, info = env.reset()
        eps_rew = 0
        for i in range(N_STEPS):
            # eps-greedy action sampling
            if np.random.uniform(0,1) < epsilon:
                a = np.random.choice(n_actions)
            else:
                a = torch.argmax(online_qnet(torch.from_numpy(s).to(device))).item()
            next_s, rew, done, trunc, info = env.step(a)
            exp_replay.insert(s, next_s, a, rew, done)
            # learn 
            if exp_replay.size() > BATCH_SIZE:
                batch_s, batch_ns, batch_a, batch_r, batch_d = exp_replay.sample_experience(BATCH_SIZE, device)
                # update target net
                if i % UPDATE_STEPS == 0:
                    target_qnet.load_state_dict(online_qnet.state_dict())
                
                q_pred = online_qnet.forward(batch_s)
                with torch.no_grad():
                    q_targ_ns = target_qnet.forward(batch_ns)
                    q_targ = torch.add(batch_r, GAMMA * (1 - batch_d) * torch.max(q_targ_ns, dim=1, keepdim=True)[0])

                loss = online_qnet.loss_func(q_pred.gather(dim=1, index=batch_a), q_targ).to(device)
                online_qnet.optimizer.zero_grad()
                loss.backward()
                online_qnet.optimizer.step()

                s = next_s
            eps_rew += rew
            if done:
                break
            if epsilon > FIN_EPS:
                epsilon -= (INIT_EPS - FIN_EPS) / EXPLORE
        if epoch % 10 == 0:
            print('Eps. Rew.:', eps_rew)





if __name__ == '__main__':
    main()