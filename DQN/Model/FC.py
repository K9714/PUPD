from DQN.Model.Agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim

class FC(Agent):
    def __init__(self, conf: dict, device):
        num_states = conf['Training']['num_states']
        num_actions = conf['Training']['num_actions']
        lr = conf['HParams']['lr']
        model1 = nn.Sequential(
            nn.Linear(num_states, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        ).to(device)
        optimizer1 = optim.Adam(model1.parameters(), lr)

        model2 = nn.Sequential(
            nn.Linear(num_states, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        ).to(device)
        optimizer2 = optim.Adam(model2.parameters(), lr)
        super().__init__(conf['HParams'], device, models=[model1, model2], optimizers=[optimizer1, optimizer2])
