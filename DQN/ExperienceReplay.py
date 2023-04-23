import random
import torch
import numpy as np
from collections import deque

class ER():
    def __init__(self, device, batch_size, maxlen=100000):
        self.device = device
        self.batch_size = batch_size
        self.memories = deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.memories)

    def sample(self):
        batch = random.sample(self.memories, self.batch_size)
        return batch
    
    def memorize(self, state, action, reward, next_state, done):
        self.memories.append((
            state,
            torch.tensor(np.array([action]), dtype=torch.int64).to(self.device),
            torch.tensor(np.array([reward]), dtype=torch.float).to(self.device),
            torch.tensor(np.array([next_state]), dtype=torch.float).to(self.device),
            torch.tensor(np.array([done]), dtype=torch.bool).to(self.device),
        ))