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

thicc = 5000

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.infc = nn.Linear(539, thicc)
    self.innorm = nn.BatchNorm1d(thicc)
    self.inact = nn.LeakyReLU()
    self.fc1 = nn.Linear(thicc, thicc)
    self.norm1 = nn.BatchNorm1d(thicc)
    self.act1 = nn.LeakyReLU()
    self.fc2 = nn.Linear(thicc, thicc)
    self.norm2 = nn.BatchNorm1d(thicc)
    self.act2 = nn.LeakyReLU()
    self.fc3 = nn.Linear(thicc, thicc)
    self.norm3 = nn.BatchNorm1d(thicc)
    self.act3 = nn.LeakyReLU()
    self.fc4 = nn.Linear(thicc, thicc)
    self.norm4 = nn.BatchNorm1d(thicc)
    self.act4 = nn.LeakyReLU()

    self.fc5 = nn.Linear(thicc, thicc)
    self.norm5 = nn.BatchNorm1d(thicc)
    self.act5 = nn.LeakyReLU()
    self.fc6 = nn.Linear(thicc, thicc)
    self.norm6 = nn.BatchNorm1d(thicc)
    self.act6 = nn.LeakyReLU()
    self.fc7 = nn.Linear(thicc, thicc)
    self.norm7 = nn.BatchNorm1d(thicc)
    self.act7 = nn.LeakyReLU()
    self.fc8 = nn.Linear(thicc, thicc)
    self.norm8 = nn.BatchNorm1d(thicc)
    self.act8 = nn.LeakyReLU()
    self.fc9 = nn.Linear(thicc, thicc)
    self.norm9 = nn.BatchNorm1d(thicc)
    self.act9 = nn.LeakyReLU()

    self.outfc = nn.Linear(thicc, 1)
  def forward(self, x):
    x = self.inact(self.innorm(self.infc(x)))
    x = self.act1(self.norm1(x + self.fc1(x)))
    x = self.act2(self.norm2(x + self.fc2(x)))
    x = self.act3(self.norm3(x + self.fc3(x)))
    x = self.act4(self.norm4(x + self.fc4(x)))
    x = self.act5(self.norm5(x + self.fc5(x)))
    x = self.act6(self.norm6(x + self.fc6(x)))
    x = self.act7(self.norm7(x + self.fc7(x)))
    x = self.act8(self.norm8(x + self.fc8(x)))
    x = self.act9(self.norm9(x + self.fc9(x)))
    x = self.outfc(x)
    return x
    

model = MLP()
print(sum(p.numel() for p in model.parameters()))

# print(model)

PATH = './epoch35-save.pth'
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
