import torch
import numpy as np

class ExperienceReplay():
    def __init__(self, max_buffer_size, n_states, is_atari=False):
        self.max_buffer_size = max_buffer_size
        self.counter = 0
        self.idx = 0

        if is_atari:
            self.state_mem = np.zeros((max_buffer_size, ) + n_states, dtype=np.float32)
            self.nstate_mem = np.zeros((max_buffer_size, ) + n_states, dtype=np.float32)
        else:
            self.state_mem = np.zeros((max_buffer_size, n_states), dtype=np.float32)
            self.nstate_mem = np.zeros((max_buffer_size, n_states), dtype=np.float32)
        self.action_mem = np.zeros(max_buffer_size, dtype=np.float32)
        self.reward_mem = np.zeros(max_buffer_size, dtype=np.float32)
        self.done_mem = np.zeros(max_buffer_size, dtype=np.float32)

    def insert(self, state, n_state, action, reward, done):
        self.idx = self.counter % self.max_buffer_size
        
        self.state_mem[self.idx] = state
        self.nstate_mem[self.idx] = n_state
        self.action_mem[self.idx] = action
        self.reward_mem[self.idx] = reward
        self.done_mem[self.idx] = done

        self.counter += 1
    
    def get_sample_idxs(self, batch_size):
        return np.random.choice(self.size(), batch_size)
    
    def sample_experience(self, batch_size, device):    
        idxs = self.get_sample_idxs(batch_size)

        batch_s = torch.tensor(self.state_mem[idxs]).to(device)
        batch_ns = torch.tensor(self.nstate_mem[idxs]).to(device)
        batch_a = self.process_vec(self.action_mem[idxs]).to(device)
        batch_r = self.process_vec(self.reward_mem[idxs]).to(device)
        batch_d = self.process_vec(self.done_mem[idxs]).to(device)
        
        return batch_s, batch_ns, batch_a, batch_r, batch_d, idxs, None

    def process_vec(self, vec):
        return torch.tensor(vec).unsqueeze(0).t().long()

    def size(self):
        return min(self.counter, self.max_buffer_size)

class PrioritizedExperienceReplay(ExperienceReplay):
    """
    Implementation of Prioritized Experience Replay (PER) using segment trees.
    PER samples transitions from the replay buffer proportional to their TD-error.
    New samples are given maximal priority to guarantee they are sampled at least once.

    In order to offset the bias introduced from sampling transitions from the replay buffer
    non-uniformly, importance sampling weights are computed and used. Weights are also then
    normalized by the maximal IS weight. 

    To compute the IS weights and max IS weight efficiently, 2 arrays are used (as segment 
    trees) to store the sum and min of the priorities. This allows O(1) extract and O(logN) 
    updating of priorities.
    """
    def __init__(self, max_buffer_size, n_states, alpha=0.2, beta=0.6, is_atari=False):
        super(PrioritizedExperienceReplay, self).__init__(max_buffer_size, n_states, is_atari)
        self.max_prio = 1
        self.alpha = alpha
        self.beta = beta
        # N leaf nodes means 2N-1 tree space. We initialize 2N space here and ignore 
        # the first index for this reason and also for convenience of tree-indexing.
        # Root node is at idx = 1
        self.sum_tree = np.zeros(self.max_buffer_size*2)
        self.min_tree = np.full(self.max_buffer_size*2, np.inf)

    def insert(self, state, n_state, action, reward, done):
        # Insert into buffer
        super().insert(state, n_state, action, reward, done)
        # Update sum and min trees
        prio = self.max_prio**self.alpha
        self.update_sum_min_tree(prio)
    
    def update_sum_min_tree(self, prio):
        node_idx = self.idx + self.max_buffer_size
        self.sum_tree[node_idx] = prio
        self.min_tree[node_idx] = prio

        while node_idx > 1:
            node_idx = node_idx//2
            self.sum_tree[node_idx] = sum(self.sum_tree[2*node_idx], self.sum_tree[2*node_idx+1])
            self.min_tree[node_idx] = min(self.min_tree[2*node_idx], self.min_tree[2*node_idx+1])
        return

    def get_sample_idxs(self, batch_size):
        probs = np.random.random(batch_size)*self.sum_tree[1]
        idxs = np.zeros(batch_size)
        for batch_i in range(batch_size):
            # Run binary search to find corresponding leaf node in tree
            i = 1
            while i < self.max_buffer_size*2:
                if self.sum_tree[i] > probs[batch_i]:
                    i = i*2
                else:
                    i = i*2+1
            idxs[batch_i] = i
        return idxs

    def sample_experience(self, batch_size, device):
        batch_s, batch_ns, batch_a, batch_r, batch_d, idxs, _ = super().sample_experience(batch_size, device)
        # Compute max IS weight for normalization
        min_prob = self.min_tree[1]/self.sum_tree[1]
        max_isw = (1/(self.size()*min_prob))**self.beta
        # Compute IS weights for batch
        weights = np.zeros(batch_size)
        for i in range(batch_size):
            leaf_idx = idxs[i] + self.max_buffer_size
            p = self.sum_tree[leaf_idx]/self.sum_tree[1]
            isw = (1/(self.size()*p))**self.beta
            weights[i] = isw/max_isw

        return batch_s, batch_ns, batch_a, batch_r, batch_d, idxs, weights


    def update_priorities(self, idxs, priorities):
        assert len(idxs) == len(priorities)

        for i in range(len(idxs)):
            self.max_prio = max(self.max_prio, priorities[i])
            self.update_sum_min_tree(priorities[i]**self.alpha)
        return
