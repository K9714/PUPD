import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self):
        self.model = nn.Sequential

    # 에피소드 저장 함수
    def memorize(self, state, action, reward, next_state):

    # 행동 담당 함수
    def act(self, state):

    # 학습 함수
    def learn(self):