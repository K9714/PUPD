import random
import torch
import numpy as np
from collections import deque
import pickle

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
            np.array(action),
            np.array(reward),
            np.array(next_state, dtype='i1'),
            np.array(done)
        ))

    def save(self, path, filename):
        with open(path + "/" + filename, "wb") as f:
            pickle.dump(self.memories, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path, filename):
        with open(path + "/" + filename, "rb") as f:
            self.memories = pickle.load(f)
