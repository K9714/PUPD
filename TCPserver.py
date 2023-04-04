import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
from DQN.Model.FC import FC
from DQN.Model.CNN import CNNModel
from DQN.Model.Agent import Agent

import socket
from _thread import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"

CONFIG_PATH = "Config/"
CONFIG_NAME = "config.yaml"

config = None

def load_config():
    global config
    with open(BASE_DIR + CONFIG_PATH + CONFIG_NAME, encoding='UTF8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    assert config != None, "Failed to load 'config.yaml' !!"

load_config()

# 통신 정보 설정
IP = 'localhost' # 192.168.0.13
PORT = 5050
SIZE = 2048
ADDR = (IP, PORT)

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
model = CNNModel(config['Training']['num_states']).to(device)
optimizer = optim.Adam(model.parameters(), config['HParams']['lr'])
agent = Agent(config['HParams'], device, models=[model], optimizers=[optimizer])

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
    frame_skip_count = 0
    frame_skip = int(config['HParams']['frame_skip'])
    while True:
        try:
            length = recvall(sock, 16)
            if not length:
                break
            data = recvall(sock, int(length.decode()))
            state = np.frombuffer(data, dtype='uint8')
            state = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            action, ret, idx = agent.action(state)
            
            send_action = str(action.item())
            data = str(len(action)).ljust(16)
            sock.send(data.encode())
            sock.send(send_action.encode())

            length = recvall(sock, 16)
            data = recvall(sock, int(length.decode()))
            reward = int(data.decode())

            length = recvall(sock, 16)
            data = recvall(sock, int(length.decode()))
            next_state = np.frombuffer(data, dtype='uint8')

            frame_skip_count += 1
            if frame_skip_count == frame_skip:
                agent.memorize(0, state, action, reward, next_state, 0)
                agent.learn(0)
                frame_skip_count = 0
            

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


