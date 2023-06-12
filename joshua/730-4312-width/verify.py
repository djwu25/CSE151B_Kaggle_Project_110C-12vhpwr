#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Get training device
if torch.cuda.is_available():
#  print("GPU is available, using device 1")
  dev = "cuda:0" 
else:  
#  print("GPU is not available, using CPU")
  dev = "cpu"  
device = torch.device(dev)

thicc = 2156 * 2

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(539, thicc)
    self.norm1 = nn.BatchNorm1d(thicc)
    self.act1 = nn.ReLU()
    self.fc2 = nn.Linear(thicc, thicc)
    self.norm2 = nn.BatchNorm1d(thicc)
    self.act2 = nn.ReLU()
    self.fc3 = nn.Linear(thicc, thicc)
    self.norm3 = nn.BatchNorm1d(thicc)
    self.act3 = nn.ReLU()
    self.fc4 = nn.Linear(thicc, thicc)
    self.norm4 = nn.BatchNorm1d(thicc)
    self.act4 = nn.ReLU()
    self.fc5 = nn.Linear(thicc, 1)
  def forward(self, x):
    x = self.act1(self.norm1(self.fc1(x)))
    x = self.act2(self.norm2(self.fc2(x)))
    x = self.act3(self.norm3(self.fc3(x)))
    x = self.act4(self.norm4(self.fc4(x)))
    x = self.fc5(x)
    return x
    

model = MLP()

print(sum(p.numel() for p in model.parameters()))

# print(model)

PATH = './save.pth'
model.load_state_dict(torch.load(PATH))
model.to(device)

csv_file = 'jtan-onehot-test.csv'
df = pd.read_csv(csv_file, dtype=np.float32)
all_xy = df.to_numpy()
taxi_inputs = torch.tensor(all_xy[:,0:539], dtype=torch.float32).to(device)

#print(taxi_inputs)

with torch.no_grad():
  outputs = model(taxi_inputs)

ids = open('tripid.txt')
print('"TRIP_ID","TRAVEL_TIME"')
for i in outputs:
  print(ids.readline().strip() + "," + str(i.item()))
