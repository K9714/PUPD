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
        optimizer1 = optim.Adam(model1.parameters(), lr)
        optimizer2 = optim.Adam(model2.parameters(), lr)
        super().__init__(conf['HParams'], device, models=[model1, model2], optimizers=[optimizer1, optimizer2])

class DuelingCNN(nn.Module):
    def __init__(self, num_actions):
        super(DuelingCNN, self).__init__()
        self.cnn_layer = nn.Sequential(
            # input 84x95x4장
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=10, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1*11*11, 512),#1*20*20, 256),
            nn.ReLU(),
            nn.Linear(512, 256),#1*20*20, 256),
            nn.ReLU()
        )
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, num_actions)

    def forward(self, x):
        x = x.view((-1, 4, 41, 40))
        x = self.cnn_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        value = self.value(x)
        adv = self.advantage(x)

        advantage_avg = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advantage_avg

        return Q

    def select_action_target(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_idx = torch.argmax(Q, dim=1)
        return action_idx.item()