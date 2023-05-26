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

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(539, 777)
    self.drop1 = nn.Dropout()
    self.norm1 = nn.BatchNorm1d(777)
    self.relu1 = nn.LeakyReLU()
    self.fc2 = nn.Linear(777, 777)
    self.drop2 = nn.Dropout()
    self.norm2 = nn.BatchNorm1d(777)
    self.relu2 = nn.LeakyReLU()
    self.fc3 = nn.Linear(777, 777)
    self.drop3 = nn.Dropout()
    self.norm3 = nn.BatchNorm1d(777)
    self.relu3 = nn.LeakyReLU()
    self.fc4 = nn.Linear(777, 777)
    self.drop4 = nn.Dropout()
    self.norm4 = nn.BatchNorm1d(777)
    self.relu4 = nn.LeakyReLU()
    self.fc5 = nn.Linear(777, 777)
    self.drop5 = nn.Dropout()
    self.norm5 = nn.BatchNorm1d(777)
    self.relu5 = nn.LeakyReLU()
    self.fc6 = nn.Linear(777, 777)
    self.drop6 = nn.Dropout()
    self.norm6 = nn.BatchNorm1d(777)
    self.relu6 = nn.LeakyReLU()
    self.fc7 = nn.Linear(777, 777)
    self.drop7 = nn.Dropout()
    self.norm7 = nn.BatchNorm1d(777)
    self.relu7 = nn.LeakyReLU()
    self.fc8 = nn.Linear(777, 777)
    self.drop8 = nn.Dropout()
    self.norm8 = nn.BatchNorm1d(777)
    self.relu8 = nn.LeakyReLU()
    self.fc9 = nn.Linear(777, 777)
    self.drop9 = nn.Dropout()
    self.norm9 = nn.BatchNorm1d(777)
    self.relu9 = nn.LeakyReLU()
    self.fc10 = nn.Linear(777, 1)
  def forward(self, x):
    x = self.drop1(self.relu1(self.norm1(self.fc1(x))))
    x = self.drop2(self.relu2(self.norm2(self.fc2(x))))
    x = self.drop3(self.relu3(self.norm3(self.fc3(x))))
    x = self.drop4(self.relu4(self.norm4(self.fc4(x))))
    x = self.drop5(self.relu5(self.norm5(self.fc5(x))))
    x = self.drop6(self.relu6(self.norm6(self.fc6(x))))
    x = self.drop7(self.relu7(self.norm7(self.fc7(x))))
    x = self.drop8(self.relu8(self.norm8(self.fc8(x))))
    x = self.drop9(self.relu9(self.norm9(self.fc9(x))))
    x = self.fc10(x)
    return x

model = MLP()

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
