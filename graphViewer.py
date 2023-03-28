import torch
import random

t = torch.range(0, 64*4-1).reshape(64, -1)

idx = [int(random.random() * 4) for _ in range(64)]
idx = torch.tensor(idx).reshape(64, 1)

print(t.shape)
print(idx.shape)


result = t.gather(1, idx)
print(result)