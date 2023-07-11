import torch
import numpy as np

class ExperienceReplay():
    def __init__(self, max_buffer_size, n_states, is_atari=False):
        self.max_buffer_size = max_buffer_size
        self.counter = 0

        if is_atari:
            self.state_mem = np.zeros((max_buffer_size, ) + n_states, dtype=np.float32)
            self.nstate_mem = np.zeros((max_buffer_size, ) + n_states, dtype=np.float32)
        else:
            self.state_mem = np.zeros((max_buffer_size, n_states), dtype=np.float32)
            self.nstate_mem = np.zeros((max_buffer_size, n_states), dtype=np.float32)
        self.action_mem = np.zeros(max_buffer_size, dtype=np.float32)
        self.reward_mem = np.zeros(max_buffer_size, dtype=np.float32)
        self.done_mem = np.zeros(max_buffer_size, dtype=np.float32)

    def insert(self, state, next_state, action, reward, done):
        idx = self.counter % self.max_buffer_size
        
        self.state_mem[idx] = state
        self.nstate_mem[idx] = next_state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.done_mem[idx] = done

        self.counter += 1
        
    def sample_experience(self, batch_size, device):    
        idxs = np.random.choice(min(self.counter, self.max_buffer_size), batch_size)

        batch_s = torch.tensor(self.state_mem[idxs]).to(device)
        batch_ns = torch.tensor(self.nstate_mem[idxs]).to(device)
        batch_a = self.process_vec(self.action_mem[idxs]).to(device)
        batch_r = self.process_vec(self.reward_mem[idxs]).to(device)
        batch_d = self.process_vec(self.done_mem[idxs]).to(device)
        
        return batch_s, batch_ns, batch_a, batch_r, batch_d

    def process_vec(self, vec):
        return torch.tensor(vec).unsqueeze(0).t().long()

class PrioritizedExperienceReplay():
    def __init__(self):
        return