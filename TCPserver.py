import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
from DQN.Model.FC import FC
from DQN.Model.CNN import CNNModel
from DQN.Model.DDQN import DDQN
from DQN.Model.Agent import Agent
from PIL import Image

import socket
from _thread import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"

CONFIG_PATH = "Config/"
CONFIG_NAME = "config.yaml"

config = None
lock = allocate_lock()

def load_config():
    global config
    with open(BASE_DIR + CONFIG_PATH + CONFIG_NAME, encoding='UTF8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    assert config != None, "Failed to load 'config.yaml' !!"

load_config()

# 통신 정보 설정
IP = '0.0.0.0' # 192.168.0.13
PORT = 5050
SIZE = 2048
ADDR = (IP, PORT)

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
# main_model = DuelingCNN(config['Training']['num_states']).to(device)
# target_model = DuelingCNN(config['Training']['num_states']).to(device)
# optimizer = optim.Adam(model.parameters(), config['HParams']['lr'])
agent = DDQN(config, device)
#agent.load("./Data/", "230407_DDDQN_eps_1340.pt")

client_sockets = []


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def client_thread(sock, addr):
    print(f">> Connected by : {addr[0]}:{addr[1]}")
    frame_stack = []
    next_frame_stack = []
    frame_skip = int(config['HParams']['frame_skip'])
    step = 0
    memorize = False
    while True:
        try:
            length = recvall(sock, 16)
            if not length:
                break
            data = recvall(sock, int(length.decode()))
            stack = np.reshape(np.frombuffer(data, dtype='uint8'), (41, 40))
            stack = stack.tolist()

            for i in range(step % frame_skip, len(frame_stack), frame_skip):
                frame_stack[i].append(stack)
            frame_stack.append([stack])

            if len(frame_stack[0]) == frame_skip:
                state = np.array([frame_stack.pop(0)])
                state = torch.tensor(state, dtype=torch.float).to(device)
                action = agent.action(state)
                memorize = True
            else:
                action = 4
                memorize = False
            
            send_action = str(action)
            data = str(len(send_action)).ljust(16)
            sock.send(data.encode())
            sock.send(send_action.encode())

            length = recvall(sock, 16)
            data = recvall(sock, int(length.decode()))
            reward = int(data.decode())

            length = recvall(sock, 16)
            data = recvall(sock, int(length.decode()))
            next_stack = np.reshape(np.frombuffer(data, dtype='uint8'), (41, 40))
            next_stack = next_stack.tolist()

            for i in range(step % frame_skip, len(next_frame_stack), frame_skip):
                next_frame_stack[i].append(next_stack)
            next_frame_stack.append([next_stack])

            if memorize:
                next_state = next_frame_stack.pop(0)
                agent.memorize(0, state, action, reward, next_state, 0)
                lock.acquire()
                agent.learn(0)
                lock.release()
                step -= 1
            step += 1

        except ConnectionResetError as e:
            break
    print(f">> Disconnect Client by : {addr[0]}:{addr[1]}")
    if sock in client_sockets:
        client_sockets.remove(sock)



# 서버 소켓 설정
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind(ADDR)  # 주소 바인딩
    server_socket.listen()  # 클라이언트의 요청을 받을 준비
    print(">> Socket Server Started!")
    # 무한루프 진입
    try:
        while True:
            client_socket, client_addr = server_socket.accept()  # 수신대기, 접속한 클라이언트 정보 (소켓, 주소) 반환
            client_sockets.append(client_socket)
            start_new_thread(client_thread, (client_socket, client_addr))        
    except Exception as e:
        print("ERROR :", e)
    finally:
        server_socket.close()


