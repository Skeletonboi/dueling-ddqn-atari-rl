import torch
import numpy as np
from collections import deque

class ExperienceReplay():
    def __init__(self, max_buffer_size):
        self.buffer = deque(maxlen=max_buffer_size)
    
    def size(self):
        return len(self.buffer)

    def insert(self, state, next_state, action, reward, done):
        self.buffer.append((state, next_state, action, reward, done))

    def sample_experience(self, batch_size, device):    
        idxs = np.random.choice(self.size(), batch_size)
        batch_s, batch_ns, batch_a, batch_r, batch_d = zip(*[self.buffer[i] for i in idxs])

        batch_s = torch.tensor(batch_s).to(device)
        batch_ns = torch.tensor(batch_ns).to(device)
        batch_a = self.process_vec(batch_a).to(device)
        batch_r = self.process_vec(batch_r).to(device)
        batch_d = self.process_vec(batch_d).to(device)

        return batch_s, batch_ns, batch_a, batch_r, batch_d

    def process_vec(self, vec):
        return torch.tensor(vec).unsqueeze(0).t().long()

class PrioritizedExperienceReplay():
    def __init__(self):
        return