from DQN.Model.Agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim

class FC(Agent):
    def __init__(self, conf: dict, device):
        num_states = conf['Training']['num_states']
        num_actions = conf['Training']['num_actions']
        lr = conf['HParams']['lr']
        model1 = FCModel(num_states, num_actions).to(device)
        #model2 = FCModel(num_states, num_actions).to(device)
        optimizer = optim.Adam(model1.parameters(), lr)
        super().__init__(conf, device, models=[model1], optimizer=optimizer)

        
class FCModel(nn.Module):
    def __init__(self, num_states, num_actions):
        super(FCModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_states, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
    def select_action_target(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_idx = torch.argmax(Q, dim=1)
        return action_idx.item(), Q
    

class DuelingFC(Agent):
    def __init__(self, conf: dict, device):
        num_states = conf['Training']['num_states']
        num_actions = conf['Training']['num_actions']
        lr = conf['HParams']['lr']
        model1 = DuelingFCModel(num_states, num_actions).to(device)
        model2 = DuelingFCModel(num_states, num_actions).to(device)
        optimizer = optim.Adam(model1.parameters(), lr)
        super().__init__(conf, device, models=[model1, model2], optimizer=optimizer)


class DuelingFCModel(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DuelingFCModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_states, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.adv = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)

        value = self.value(x)
        adv = self.adv(x)

        advantage_avg = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advantage_avg

        return Q

    def select_action_target(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_idx = torch.argmax(Q, dim=1)
        return action_idx.item(), Q