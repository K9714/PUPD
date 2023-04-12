from DQN.Model.Agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim

class CNN(Agent):
    def __init__(self, conf: dict, device):
        num_states = conf['Training']['num_states']
        num_actions = conf['Training']['num_actions']
        lr = conf['HParams']['lr']
        model1 = CNNModel(num_actions).to(device)
        optimizer1 = optim.Adam(model1.parameters(), lr)
        super().__init__(conf['HParams'], device, models=[model1], optimizers=[optimizer1])

class CNNModel(nn.Module):
    def __init__(self, num_actions):
        super(CNNModel, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=10, stride=2),
            nn.ReLU(),
            # 1 x 176 x 176
            nn.MaxPool2d(2),
            # 1 x 88 x 88
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=10, stride=2),
            nn.ReLU(),
            # 1 x 40 x 40
            nn.MaxPool2d(2),
            # 1 x 20 x 20
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(460, 256),#1*20*20, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        x = x.view((-1, 1, 360, 410))
        x = self.cnn_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x