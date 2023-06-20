import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
import pickle
import numpy as np

from PIL import Image
import random
# (score, runtime, total_reward, avg_loss)
class DataLoader():
    def save(self, path, data):
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
    def __init__(self, path, path2=None, use_pickle=False):
        a = 0.3
        b = (1 - a)

        self.score = []
        self.runtime = []
        self.total_reward = []
        self.avg_loss = []
        self.spt = []
        self.ev_score = []
        self.fc_data = []
        self.eps = []

        if not use_pickle:
            data = torch.load(path)
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            self.fc_data = data
                
            data = {'score_hist': data}
        for i, hist in enumerate(data['score_hist']):
            self.score.append(hist[0])
            self.runtime.append(hist[1])
            self.spt.append(hist[0] / hist[1])
            self.ev_score.append(a * self.runtime[-1] + b * self.spt[-1])
            self.total_reward.append(hist[2])
            self.avg_loss.append(hist[3])

        if path2 is not None:
            data  = torch.load(path2)
            for hist in data['score_hist']:
                self.score.append(hist[0])
                self.runtime.append(hist[1])
                self.spt.append(hist[0] / hist[1])
                self.ev_score.append(a * self.runtime[-1] + b * self.spt[-1])
                self.total_reward.append(hist[2])
                self.avg_loss.append(hist[3])

        for i in range(0, 1500, 20):
            self.eps.append(sum(self.score[i:i+20]) / 20)


path = {
    "GRAY1": "./Data/DDQN_score_history_eps_1601.pt",
}

data = {}
plt.figure(figsize=(10,5))
plt.xlabel("Episode", fontdict={'size': 12})
plt.ylabel("Evaluation", fontdict={'size': 12})
for k, v in path.items():
    if type(v) == list:
        data[k] = DataLoader(v[0], v[1])
    elif type(v) == tuple:
        data[k] = DataLoader(v[0], use_pickle=True)
    else:
        data[k] = DataLoader(v)
    if k == "GRAY1":
        target = data[k].score
        plt.plot(target, color='blue', label=k, zorder=1)
        avg = sum(target) / len(target)
        plt.hlines(avg, 0, len(target), colors='red', linestyle='--', linewidth=2, zorder=2, label=f"Avg: {round(avg,2)}")

    print(f"{k} avg : {sum(data[k].ev_score) / len(data[k].ev_score)}")

plt.legend(loc='upper right')
plt.show()