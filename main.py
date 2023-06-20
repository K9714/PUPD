import os
import yaml
import torch
import time
import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import multiprocessing
from DQN.Env import Env
from Util.Image import *
from Util.Memory import rwm
import Util.Process as Process

from torchsummary import summary
from DQN.Model.FC import FC, DuelingFC
from DQN.Model.DDQN import CNN, DuelingDDQN
from DQN.FrameProcessor import FrameProcessor
import torch.nn as nn

import torch
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"

CONFIG_PATH = "Config/"
CONFIG_NAME = "config.yaml"

# Config(Config/config.yaml)
config = None
# Pinball process (Util.Process)
proc = None
# Pinball handle
handle = None
# Pinball score memory address
score_addr = None

def load_config():
    global config
    with open(BASE_DIR + CONFIG_PATH + CONFIG_NAME, encoding='UTF8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    assert config != None, "Failed to load 'config.yaml' !!"

def init():
    global config, proc, handle, score_addr
    load_config()
    # Start Pinball process
    proc = Process.start(BASE_DIR + config['Pinball']['path'] + config['Pinball']['file_name'], config['Pinball'])

def run():
    global config, proc, handle, score_addr

    board = SummaryWriter()

    # Set hyper parameters
    hp = config['HParams']
    eps = hp['episodes']
    lr = hp['lr']
    batch_size = hp['batch_size']
    frame_skip = int(hp['frame_skip'])
    frame_stack = int(hp['frame_stack'])
    save_eps = config['Training']['save_eps']
    update_freq = hp['update_freq']

    # Set score history
    score_hist = []
    # Set Environment
    env = Env(BASE_DIR, config, proc)
    # Set Agent
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    #agent = DuelingFC(config, device)
    agent = CNN(config, device)
    agent.load('./Data_backup/230525_Data_CNN_DDQN_FULL', 'CNN_params_eps_801.pt')

    #model = DuelingCNN(config['Training']['num_actions'])
    #summary(model, input_size=(4, 180, 180)).to(device)

    stack = FrameProcessor(frame_skip)
    
    for e in range(1, eps + 1):
        steps = 0
        total_reward = 0
        avg_loss = 0
        frame = env.reset()
        print(f"EPS : {e}")
        start_t = time.time()
        memorize = False
        stack.reset()
        while True:
            if env.in_start:
                if not env.plunger_full:
                    action = 9
                else:
                    action = 0
                    time.sleep(1)
                next_frame, reward, done = env.step(action)
            else:
                
                stack.frame_append(frame)
                if len(stack) < frame_stack:
                    action = 0
                else:
                    state = np.array(stack.frame_pop(), dtype='i1')
                    action, Q = agent.action(state)
                    memorize = True

                next_frame, reward, done = env.step(action)
                stack.next_frame_append(next_frame)
                # 게임이 끝났을 경우 마이너스 보상주기 
                if done:
                    reward = -10
                
                if memorize:
                    next_state = stack.next_frame_pop()
                    agent.memorize(state, action, reward, next_state, done) # 경험(에피소드) 기억
                    loss = 0
                    #loss = agent.learn()
                    print(f"Action : {action}, Reward : {reward}, Loss : {round(loss, 3)}, Q-Value : {Q}")
                    #avg_loss += loss
                """
                state = np.array([frame])
                action, Q = agent.action(state)
                next_frame, reward, done = env.step(action)
                if done:
                    reward = -10
                agent.memorize(state, action, reward, next_frame, done)
                loss = 0
                #loss = agent.learn()
                #print(f"State : {state}, NextState : {next_frame}, Q-Value : {Q}")
                print(f"Action : {action}, Reward : {reward}, Loss : {round(loss, 3)}, Q-Value : {Q}")
                #avg_loss += loss
                """


            total_reward += reward
            frame = next_frame
            steps += 1
            
            if done:
                end_t = time.time()
                runtime = end_t - start_t
                score = rwm.ReadProcessMemory(env.proc.handle, env.proc.score_addr)
                print("에피소드:{0} 점수: {1}, 수행시간 : {2}".format(e, score, runtime))
                for _ in tqdm(range(int(steps / update_freq))):
                    avg_loss += agent.learn()
                #agent.scheduler.step()
                avg_loss = avg_loss / steps
                board.add_scalar("Score/Episode", score, e)
                board.add_scalar("Runtime/Episode", runtime, e)
                board.add_scalar("Total Reward/Episode", total_reward, e)
                board.add_scalar("Avg. Loss/Episode", avg_loss, e)
                evaluation = score ** 0.2 + runtime ** 0.2
                board.add_scalar("Evaluation/Episode", evaluation, e)
                board.flush()
                score_hist.append((score, runtime, total_reward, avg_loss, evaluation)) #score history에 점수 저장
                if (e - 1) % save_eps == 0:
                    name = type(agent).__name__
                    torch.save({'score_hist': score_hist}, f"./Data/{name}_score_history_eps_{e}.pt")
                    agent.save("./Data", f"{name}_params_eps_{e}.pt")

                break
    board.close()

if __name__ == "__main__":
    init()
    run()