import random
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import torch.nn.functional as F
from DQN.ExperienceReplay import ER

class Agent:
    def __init__(self, conf: dict, device, models, optimizer):

        self.num_actions = conf['Training']['num_actions']

        conf_hparams = conf['HParams']
        self.eps_start = conf_hparams['eps_start']
        self.eps_end = conf_hparams['eps_end']
        self.eps_decay = conf_hparams['eps_decay']

        self.alpha = conf_hparams['alpha']
        self.gamma = conf_hparams['gamma']
        self.tau = conf_hparams['tau']
        self.batch_size = conf_hparams['batch_size']
        self.ddqn_update_step = conf_hparams['ddqn_update_step']

        self.device = device

        self.models = models
        self.optimizer = optimizer
        #self.scheduler = CyclicLR(optimizer, base_lr=conf_hparams['lr'], max_lr=0.001, step_size_up=5, step_size_down=10, mode='triangular2', cycle_momentum=False)
        self.memory = ER(device, self.batch_size, maxlen=50000)
        self.steps_done = 0
        self.update_step = 0
        self.old_action = 0

    def memorize(self, state, action, reward, next_state, done):
        self.memory.memorize(state, action, reward, next_state, done)

    def action(self, state: list) -> tuple:
        threshold = max(self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay), self.eps_end)
        if threshold != self.eps_end:
            self.steps_done += 1
        if random.random() > threshold:
            #state = torch.tensor(state, dtype=torch.float).to(self.device)
            state = torch.tensor(state / 255., dtype=torch.float).to(self.device)
            if len(self.models) > 1:
                ret, Q = self.models[1].select_action_target(state)
            else:
                Q = self.models[0](state).detach()
                ret = Q.max(1)[1].view(1, 1).item()
            return ret, Q
        else:
            return random.randrange(0,self.num_actions), 0


    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0
        batch = self.memory.sample()
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states) / 255.
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states) / 255.
        dones = np.array(dones)
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device).squeeze()
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device).squeeze()
        
        if len(self.models) > 1:
            current = self.models[0]
            target = self.models[1]
            optimizer = self.optimizer
            
            current.train()
            target.eval()

            q_value = current(states)
            next_q_values = current(next_states)
            with torch.no_grad():
                next_q_state_values = target(next_states)

            #q_value = q_value.gather(1, actions).squeeze(1)
            q_value = q_value.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
            expected_q_value = rewards + self.gamma * next_q_value * ~(dones)
            self.optimizer.zero_grad()
            loss = F.smooth_l1_loss(q_value, expected_q_value.detach())#(q_value - expected_q_value.detach()).pow(2).mean()
        else:
            model = self.models[0]
            optimizer = self.optimizer
            
            #current_q = model(states).gather(1, actions).squeeze(1)
            current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            max_next_q = model(next_states).detach().max(1)[0]
            expected_q = rewards + self.gamma * max_next_q * ~(dones)
            self.optimizer.zero_grad()
            loss = F.smooth_l1_loss(current_q, expected_q)
        loss.backward()
        optimizer.step()

        del states
        del actions
        del rewards
        del next_states
        del dones
        torch.cuda.empty_cache()
        
        if len(self.models) > 1:
            # Soft Update
            current_model_state_dict = self.models[0].state_dict()
            target_model_state_dict = self.models[1].state_dict()
            for key in target_model_state_dict:
                target_model_state_dict[key] = current_model_state_dict[key] * self.tau + target_model_state_dict[key] * (1 - self.tau)
            self.models[1].load_state_dict(target_model_state_dict)

            #self.update_step += 1
            #if self.update_step >= self.ddqn_update_step:
            #    self.update_step = 0
            #    self.models[1].load_state_dict(self.models[0].state_dict())
            #    print(">> DDQN Target Model Update Params!!")
        return loss.item()

    def save(self, path, filename):
        save_dict = {}
        save_dict['model_len'] = len(self.models)
        for i, model in enumerate(self.models):
            save_dict[f'model_{i}_class'] = type(model).__name__
            save_dict[f"model_{i}"] = model.state_dict()
        save_dict["optimizer"] = self.optimizer.state_dict()
        save_dict['steps_done'] = self.steps_done
        #save_dict['memory'] = self.memory
        torch.save(save_dict, path + "/" + filename)

        #with open(path + "/" + f"{type(model).__name__}_ER.pickle", "wb") as f:
        #    pickle.dump(self.memory, f)

    def load(self, path, filename):
        data = torch.load(path + "/" + filename)
        for i in range(data['model_len']):
            self.models[i].load_state_dict(data[f'model_{i}'])
        self.optimizer.load_state_dict(data[f'optimizer'])
        self.steps_done = data['steps_done']
        #self.memory = data['memory']

        name = data['model_0_class']
        #with open(path + "/" + f"{name}_ER.pickle", "rb") as f:
        #    self.memory = pickle.load(f)
