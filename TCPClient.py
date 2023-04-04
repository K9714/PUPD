import os
import yaml
import torch
import socket
import time
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

def run():
    global config, proc, handle, score_addr, runtime_hist

    # Set hyper parameters
    hp = config['HParams']
    eps = hp['episodes']
    frame_skip = int(hp['frame_skip'])

    # Set score history
    score_hist = []
    # Set Environment
    env = Env(BASE_DIR, config, proc)
    IP = '127.0.0.1'#'192.168.0.13'
    PORT = 5050

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((IP, PORT))
    while True:
        try:
            for e in range(1, eps + 1):
                steps = 0
                state = env.reset()
                print(f"EPS : {e}")
                start_t = time.time()
                frame_skip_count = 0
                while True:
                    if env.in_start and not env.plunger_full:
                        action = 3
                        frame, next_state, reward, done = env.step(action)
                    else:
                        state = state.tobytes()
                        data = str(len(state)).ljust(16)
                        sock.send(data.encode())
                        sock.send(state)
                        length = recvall(sock, 16)
                        data = recvall(sock, int(length.decode()))
                        action = int(data.decode())

                        frame, next_state, reward, done = env.step(action)
                        print("Reward :", reward)
                        send_reward = str(reward)
                        data = str(len(send_reward)).ljust(16)
                        sock.send(data.encode())
                        sock.send(send_reward.encode())

                        send_next_state = next_state.tobytes()
                        data = str(len(send_next_state)).ljust(16)
                        sock.send(data.encode())
                        sock.send(send_next_state)

                    state = next_state
                    steps += 1 

                    if done:
                        end_t = time.time()
                        runtime = end_t - start_t
                        score = rwm.ReadProcessMemory(env.proc.handle, env.proc.score_addr)
                        print("에피소드:{0} 점수: {1}, 수행시간 : {2}".format(e, score, runtime))
                        score_hist.append([score, runtime]) #score history에 점수 저장
                        break

        except ConnectionAbortedError as e:
            print("ERROR :", e)
            break
    sock.close()



def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


init()
run()

