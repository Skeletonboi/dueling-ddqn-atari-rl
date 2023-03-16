import os, sys
sys.path.append(os.getcwd())
sys.path.append('../utils')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Stub to get SB3 working w/ gym == 0.26.0
sys.modules["gym"] = gym
# Import atari wrappers from SB3
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

import random
from tqdm import tqdm
from replay_buffers import ExperienceReplay

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import envpool

class AtariDQN(nn.Module):
    def __init__(self, input_shape, batch_size, fc_size_list, activation, lr, loss_func):
        super(AtariDQN, self).__init__()
        self.input_shape = input_shape
        self.batch_size = batch_size

        self.conv_net = self.create_conv_net(activation)
        self.conv_out_dims = self.get_conv_out_dims()
        self.fc_net = self.create_fc_net(fc_size_list, activation)

        self.loss_func = loss_func
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def create_conv_net(self, activation):
        conv_layers = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, 8, stride=4), activation,
            nn.Conv2d(32, 64, 4, stride=2), activation,
            nn.Conv2d(64, 64, 3, stride=1), activation,
            # nn.Flatten(start_dim=0)
        )
        return conv_layers

    # method slightly differs from dqn.py for conv_net inputs
    def create_fc_net(self, fc_size_list, activation):
        fc_layers = []
        for i in range(len(fc_size_list)):
            if i == 0:
                layer = nn.Linear(self.conv_out_dims, fc_size_list[i])
            else:
                layer = nn.Linear(fc_size_list[i-1], fc_size_list[i])
            if i == len(fc_size_list) - 1:
                activation = nn.Identity()
            fc_layers += ((layer, activation))
        return nn.Sequential(*fc_layers)

    def get_conv_out_dims(self):
        temp = torch.zeros(1, *self.input_shape)
        temp_out = self.conv_net(temp)
        return int(np.prod(temp_out.size()))

    def forward(self, x):
        conv_out = self.conv_net(x)
        flatten_out = conv_out.reshape(-1, self.conv_out_dims)
        q = self.fc_net(flatten_out)
        return q

def eval_model(model, env, max_steps, device):
    eps_rew = 0
    s, _ = env.reset()
    for i in range(max_steps):
        a = torch.argmax(model(torch.Tensor(np.array(s)).to(device))).item()
        next_s, rew, done, trunc, info = env.step(a)
        eps_rew += rew
        if done: break
        s = next_s
    return eps_rew

# Referenced make_env from CleanRL implementation
def make_env(env_id, seed, capture_video, run_name):
    env = gym.make(env_id)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # if capture_video:
        # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=20)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def main():
    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    
    # Hyperparameters
    N_EPOCH = int(2e4)
    N_STEPS = int(1.08e5)
    UPDATE_STEPS = 4
    TOTAL_TIMESTEPS = int(5e6)

    # N_ENVS = 16
    BATCH_SIZE = 32
    BUFFER_SIZE = int(1e5)
    UPDATE_TARGET = 16

    GAMMA = 0.99
    INIT_LR = 1e-4
    IS_LR_DECAY = False
    LR_DECAY = 0.0001
    INIT_EPS = 1.0
    FIN_EPS = 0.001
    EXPLORE = 100000
    epsilon = INIT_EPS
    lr = INIT_LR
    # Initialize env
    env = make_env("ALE/Pong-v5", SEED, False, "run1")

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    # Initialize online and target q-networks and exp. replay buffer
    online_qnet = AtariDQN(obs_shape, BATCH_SIZE, fc_size_list=[512, n_actions], activation=nn.ReLU(), lr=INIT_LR, loss_func=nn.MSELoss()).to(device)
    target_qnet = AtariDQN(obs_shape, BATCH_SIZE, fc_size_list=[512, n_actions], activation=nn.ReLU(), lr=INIT_LR, loss_func=nn.MSELoss()).to(device)
    target_qnet.load_state_dict(online_qnet.state_dict())
    exp_replay = ExperienceReplay(BUFFER_SIZE, obs_shape)

    # Main loop
    accum_rew = []
    step_counter = 0
    for epoch in tqdm(range(N_EPOCH)):
        # Train one episode
        s, info = env.reset()
        for i in range(N_STEPS):
            step_counter += 1
            # eps-greedy action sampling
            if np.random.uniform(0,1) < epsilon:
                a = np.random.choice(n_actions)
            else:
                a = torch.argmax(online_qnet(torch.Tensor(np.array(s)).to(device))).item()
            # perform action and record
            next_s, rew, done, trunc, info = env.step(a)
            exp_replay.insert(s, next_s, a, rew, done)
            # learn 
            if exp_replay.counter > BATCH_SIZE and (step_counter % UPDATE_STEPS == 0):
                batch_s, batch_ns, batch_a, batch_r, batch_d = exp_replay.sample_experience(BATCH_SIZE, device)
                # update target net
                if step_counter % UPDATE_TARGET == 0:
                    target_qnet.load_state_dict(online_qnet.state_dict())
                # compute q-target
                q_pred = online_qnet.forward(batch_s)
                with torch.no_grad():
                    q_targ_ns = target_qnet.forward(batch_ns)
                    q_targ = torch.add(batch_r, GAMMA * (1 - batch_d) * torch.max(q_targ_ns, dim=1, keepdim=True)[0])
                # compute loss and backprop
                loss = online_qnet.loss_func(q_pred.gather(dim=1, index=batch_a), q_targ).to(device)
                online_qnet.optimizer.zero_grad()
                loss.backward()
                online_qnet.optimizer.step()

                s = next_s

            if epsilon > FIN_EPS:
                epsilon -= (INIT_EPS - FIN_EPS) / EXPLORE
            if done:
                break

        # Learning rate decay scheduling
        if IS_LR_DECAY:
            lr = lr/(1 + LR_DECAY * epoch)
            for g in online_qnet.optimizer.param_groups:
                g['lr'] = lr 
        # Evaluate model pure-greedy
        eps_rew = eval_model(online_qnet, env, N_STEPS, device)
        accum_rew.append(eps_rew)
        
        if epoch % 10 == 0:
            print('Eps. Rew.:', eps_rew)
            print('Steps so far:', step_counter)
            print('Epsilon:', epsilon)
        
        if step_counter >= TOTAL_TIMESTEPS:
            break
    fig = plt.figure()
    plt.plot(accum_rew)
    plt.savefig('./imgs/atari_rew.png')
    # plt.close(fig)


if __name__ == '__main__':
    main()