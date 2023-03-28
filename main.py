import os
import yaml
import torch
import time
import pickle
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
import multiprocessing
from DQN.Env import Env
from Util.Image import *
from Util.Memory import rwm
import Util.Process as Process

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


class Animation():
    def __init__(self, data = None):
        self.hist = [0]
        self.data = data

    def animate(self, i):
        plt.cla()
        plt.plot(self.data.hist)

def animate_init(ani):
    graph = FuncAnimation(plt.gcf(), ani.animate, interval=1000)
    plt.tight_layout()
    plt.show()

from DQN.Model.FC import FC
from DQN.Model.CNN import CNN

runtime_hist = []

def run():
    global config, proc, handle, score_addr, runtime_hist

    # Set hyper parameters
    hp = config['HParams']
    eps = hp['episodes']
    lr = hp['lr']
    frame_skip = int(hp['frame_skip'])

    # Set score history
    score_hist = []
    # Set Environment
    env = Env(BASE_DIR, config, proc)
    # Set Agent
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    agent = CNN(config, device)
    

    data = Animation()
    ani = Animation(data)
    plt_proc = multiprocessing.Process(target=animate_init, args=(ani,))
    plt_proc.start()

    for e in range(1, eps + 1):
        steps = 0
        state = env.reset()
        print(f"EPS : {e}")
        start_t = time.time()
        frame_skip_count = 0
        while True:
            state = torch.tensor([state], dtype=torch.float).to(device)
            action, ret, idx = agent.action(state)
            if env.in_start and not env.plunger_full:
                action = torch.tensor([[3]]).to(device)
                

            print(action.tolist(), ret, end=" ")
            frame, next_state, reward, done = env.step(action.tolist()[0][0])
            print(reward)

            # 게임이 끝났을 경우 마이너스 보상주기 
            if done:
                reward = -5
            if not env.in_start:
                frame_skip_count += 1
                if frame_skip_count == frame_skip or done:
                    agent.memorize(frame, state, action, reward, next_state, idx) # 경험(에피소드) 기억
                    agent.learn(idx)
                    frame_skip_count = 0

            state = next_state
            steps += 1 

            if done:
                end_t = time.time()
                runtime = end_t - start_t
                score = rwm.ReadProcessMemory(env.proc.handle, env.proc.score_addr)
                print("에피소드:{0} 점수: {1}, 수행시간 : {2}".format(e, score, runtime))
                score_hist.append([score, runtime]) #score history에 점수 저장
                runtime_hist.append(runtime)
                if (e - 1) % 100 == 0:
                    torch.save({'score_hist': score_hist}, "./Data/score_history.pt")
                    agent.save("./Data/", f"eps_{e}.pt")
                break

if __name__ == "__main__":
    init()
    run()