from DQN.Model.Agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim

class DDQN(Agent):
    def __init__(self, conf: dict, device):
        num_states = conf['Training']['num_states']
        num_actions = conf['Training']['num_actions']
        lr = conf['HParams']['lr']
        model1 = DuelingCNN(num_actions).to(device)
        model2 = DuelingCNN(num_actions).to(device)
        optimizer = optim.Adam(model1.parameters(), lr)
        super().__init__(conf['HParams'], device, models=[model1, model2], optimizer=optimizer)

class DuelingCNN(nn.Module):
    def __init__(self, num_actions):
        super(DuelingCNN, self).__init__()
        self.cnn_layer = nn.Sequential(
            # input 84x95x4장
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(3*3*64, 512),#1*20*20, 256),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.adv = nn.Sequential(
            nn.Linear(3*3*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x.view((-1, 4, 180, 180))
        x = self.cnn_layer(x)
        x = x.view(x.size(0), -1)

        value = self.value(x)
        adv = self.adv(x)

        advantage_avg = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advantage_avg

        return Q

    def select_action_target(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_idx = torch.argmax(Q, dim=1)
        return action_idx.item()