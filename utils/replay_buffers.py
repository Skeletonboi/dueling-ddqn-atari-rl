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

    def sample_experience(self, batch_size):    
        idxs = np.random.choice(min(batch_size, len(self.buffer)), batch_size)
        batch_s, batch_ns, batch_a, batch_r, batch_done = zip(*[self.buffer[i] for i in idxs])
        return batch_s, batch_ns, batch_a, batch_r, batch_done

class PrioritizedExperienceReplay():
    def __init__(self):
        return