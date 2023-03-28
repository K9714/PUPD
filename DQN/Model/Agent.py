
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

class Agent:
    def __init__(self, conf_hparams: dict, device, models, optimizers):

        self.eps_start = conf_hparams['eps_start']
        self.eps_end = conf_hparams['eps_end']
        self.eps_decay = conf_hparams['eps_decay']

        self.alpha = conf_hparams['alpha']
        self.gamma = conf_hparams['gamma']
        self.batch_size = conf_hparams['batch_size']

        self.device = device

        self.models = models
        self.optimizers = optimizers
        self.memories: list = [deque([], maxlen=1000) for _ in range(len(self.models))]

        self.steps_done = 0

    def memorize(self, frame, state, action, reward, next_state, idx):
        self.memories[idx].append((
            frame,
            state,
            action,
            torch.tensor([reward]).to(self.device),
            torch.tensor([next_state], dtype=torch.float).to(self.device)
        ))

    def action(self, state: list) -> tuple:
        threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > threshold:
            idx = random.randint(0, len(self.models)-1)
            ret = self.models[idx](state)
            return ret.detach().max(1)[1].view(1, 1), ret.tolist(), idx
        else:
            return torch.tensor([[random.randrange(0,7)]]).to(self.device), [], random.randint(0, len(self.models)-1)


    def learn(self, idx):
        if len(self.memories[idx]) < self.batch_size:
            return
        batch = random.sample(self.memories[idx], self.batch_size)
        frame, states, actions, rewards, next_states = zip(*batch)
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        model = self.models[idx]
        optimizer = self.optimizers[idx]
        
        current_q = model(states).gather(1, actions) * (1 - self.alpha)
        max_next_q = model(next_states).detach().max(1)[0]
        expected_q = self.alpha * (rewards + (self.gamma * max_next_q))
        #print(current_q.squeeze(), expected_q)
        loss = F.mse_loss(current_q.squeeze(), expected_q)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def save(self, path, filename):
        save_dict = {}
        save_dict['model_len'] = len(self.models)
        for i, model in enumerate(self.models):
            save_dict[f"model_{i}"] = model.state_dict()
            save_dict[f"optimizer_{i}"] = self.optimizers[i].state_dict()
        save_dict['steps_done'] = self.steps_done
        save_dict['memories'] = self.memories

        torch.save(save_dict, path + filename)

    def load(self, path, filename):
        data = torch.load(path + filename)
        for i in range(data['model_len']):
            self.models[i].load_state_dict(data[f'model_{i}'])
            self.optimizers[i].load_state_dict(data[f'optimizer_{i}'])
        self.steps_done = data['steps_done']
        self.memories = data['memories']

