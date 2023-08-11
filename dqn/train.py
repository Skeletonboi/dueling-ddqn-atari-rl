import os, sys
import argparse
sys.path.append(os.getcwd())
sys.path.append('../utils')
import copy
import shutil
import json
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from replay_buffers import ExperienceReplay, PrioritizedExperienceReplay
from plot import plot_rewards

class DQN(nn.Module):
    def __init__(self, fc_size_list, activation, lr, loss_func, optim, device):
        super(DQN, self).__init__()
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

def eval_model(model, env, max_steps, n_actions, env_seed):
    eps_rew = 0
    s, _ = env.reset(seed=env_seed)
    for i in range(max_steps):
        a = model.sample_discrete_action(s, -1, n_actions)
        next_s, rew, done, _, _ = env.step(a)
        eps_rew += rew
        if done: break
        s = next_s
    return eps_rew

def main(args, run_path):
    # Load hyperparameters
    RUN_NAME = args['RUN_NAME']
    SEED = args['SEED']
    DDQN = args['DDQN']
    PER = args['PER']
    PER_ALPHA = args['PER_ALPHA']
    PER_BETA = args['PER_BETA']
    USE_GPU = args['USE_GPU']
    WINDOW = args['PLOT_WINDOW']

    TOTAL_TIMESTEPS = int(args['TOTAL_TIMESTEPS'])
    N_STEPS = int(args['N_STEPS'])
    UPDATE_STEPS = int(args['UPDATE_STEPS'])
    UPDATE_TARGET = int(args['UPDATE_TARGET'])

    BATCH_SIZE = int(args['BATCH_SIZE'])
    BUFFER_SIZE = int(args['BUFFER_SIZE'])

    GAMMA = float(args['GAMMA'])
    INIT_LR = float(args['INIT_LR'])
    IS_LR_DECAY = args['IS_LR_DECAY']
    LR_DECAY = float(args['LR_DECAY'])
    INIT_EPS = float(args['INIT_EPS'])
    FIN_EPS = float(args['FIN_EPS'])
    EXPLORE = int(args['EXPLORE'])

    # Set seeds
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env_seed = np.random.randint(0,100)
    
    epsilon = INIT_EPS
    lr = INIT_LR

    # Set device
    if USE_GPU: 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    # Initialize env
    # env = gym.make('CartPole-v1')
    env = gym.make("LunarLander-v2")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize online and target q-networks and exp. replay buffer
    online_qnet = DQN(fc_size_list=[n_states, 64, 64, n_actions], activation=nn.ReLU(), 
                      lr=INIT_LR, loss_func=nn.MSELoss(reduction="none"), optim=torch.optim.Adam, device=device)
    target_qnet = copy.deepcopy(online_qnet)
    target_qnet.load_state_dict(online_qnet.state_dict())
    if PER:
        exp_replay = PrioritizedExperienceReplay(BUFFER_SIZE, n_states, PER_ALPHA, PER_BETA, is_atari=False)
    else:
        exp_replay = ExperienceReplay(BUFFER_SIZE, n_states, is_atari=False)
    # Initialize counters
    accum = {'rew':[],
             'eval_rew':[],
             'steps':[]}
    epoch_counter = 0
    step_counter = 0
    lstep_counter = 0
    pbar = tqdm(total=TOTAL_TIMESTEPS)
    # Main training loop
    while step_counter < TOTAL_TIMESTEPS:
        epoch_counter += 1
        # reset environment
        s, _ = env.reset(seed=env_seed)
        eps_rew = 0
        # train one episode
        for i in range(N_STEPS):
            step_counter += 1
            pbar.update()
            # action sampling: eps-greedy
            a = online_qnet.sample_discrete_action(s, epsilon, n_actions)
            # perform action and record
            next_s, rew, done, trunc, info = env.step(a)
            exp_replay.insert(s, next_s, a, rew, done)
            eps_rew += rew
            # learn 
            if exp_replay.counter > BATCH_SIZE and (step_counter % UPDATE_STEPS == 0):
                lstep_counter += 1
                # update target net
                if lstep_counter % UPDATE_TARGET == 0:
                    target_qnet.load_state_dict(online_qnet.state_dict())
                # sample exp replay buffer
                batch_s, batch_ns, batch_a, batch_r, batch_d, batch_idx, batch_isw = exp_replay.sample_experience(BATCH_SIZE, device)
                # compute q-target
                q_pred = online_qnet.forward(batch_s)
                with torch.no_grad():
                    if DDQN:
                        # Action maximization using online qvalue of next states
                        online_q_ns = online_qnet.forward(batch_ns)
                        max_a_ns = torch.max(online_q_ns, dim=1, keepdim=True)[1]
                        # Compute q-target using online max action
                        targ_q_ns = target_qnet.forward(batch_ns)
                        q_ns = torch.gather(targ_q_ns, 1, max_a_ns)
                    else:
                        # Compute q-target using target max action
                        targ_q_ns = target_qnet.forward(batch_ns)
                        q_ns = torch.max(targ_q_ns, dim=1, keepdim=True)[0]
                        
                    q_targ = torch.add(batch_r, GAMMA * (1 - batch_d) * q_ns)
                # compute loss and backprop
                loss = online_qnet.loss_func(q_pred.gather(dim=1, index=batch_a), q_targ).to(device)
                if PER:
                    # Update PER priorities with new TD-loss
                    exp_replay.update_priorities(batch_idx, loss)
                    # Multiply importance-sampling weights
                    # print(loss)
                    # print(batch_isw)
                    loss = torch.mean(loss*batch_isw)
                else:
                    loss = torch.mean(loss)
                online_qnet.optimizer.zero_grad()
                loss.backward()
                online_qnet.optimizer.step()

            s = next_s

            # Epsilon decay scheme: linearly decreasing w.r.t. # of EXPLORE steps
            epsilon = max(FIN_EPS, epsilon - (INIT_EPS - FIN_EPS) / EXPLORE)

            if done: break
        # Learning rate decay scheduling:
        if IS_LR_DECAY:
            lr = lr/(1 + LR_DECAY * epoch_counter)
            for g in online_qnet.optimizer.param_groups:
                g['lr'] = lr 
        # Accumulate
        accum['rew'].append(eps_rew)
        accum['steps'].append(step_counter)
        if epoch_counter % WINDOW == 0:
            # Evaluate model using deterministic greedy policy
            eval_rew = eval_model(online_qnet, env, N_STEPS, n_actions, env_seed)
            accum['eval_rew'].append(eval_rew)
            rolling_rew = np.mean(accum['rew'][-WINDOW:])

            print(f'Step Counter: {step_counter}')
            print(f'Epoch Counter: {epoch_counter}')
            print(f'Epsilon: {epsilon}')
            print(f'Rolling Eps. Rew.: {rolling_rew}')
            print(f'Eval. Rew.: {eval_rew}')
            if IS_LR_DECAY:
                print(f'LR: {lr}')
            print(f'EPS: {epsilon}')

            plot_rewards(accum, WINDOW, run_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams", type=str)
    args = parser.parse_args()
    # Load hyperparameter json to dict.
    with open(f"./{args.hparams}") as f:
        hparams = json.load(f)
    
    # Create output dir and copy hyperparameters=
    run_path = f"../runs/run_{hparams['RUN_NAME']}"
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    shutil.copy(f"./{args.hparams}", run_path + f"/{args.hparams}")

    main(hparams, run_path)