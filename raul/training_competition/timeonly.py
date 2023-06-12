#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os.path

# one_hot = torch.nn.functional.one_hot(target)

# Get training device
if torch.cuda.is_available():
  print("GPU is available, using device 1")
  dev = "cuda:0" 
else:  
  print("GPU is not available, using CPU")
  dev = "cpu"  
device = torch.device(dev)

# Data class for taxi
class taxiData(Dataset):
  def __init__(self, csv_file):
    if os.path.isfile('cached_dataframe.pkl'):
      df = pd.read_pickle('cached_dataframe.pkl')
    else:
      df = pd.read_csv(csv_file)
      df.to_pickle('cached_dataframe.pkl')
    all_xy = df.to_numpy()
    
#    tmp_x = all_xy[:,0:7]
 
    call_type_data =    np.array(all_xy[:,0:66]   )
    taxi_id_data =      np.array(all_xy[:,66:516] )
    month_data =        np.array(all_xy[:,516:528])
    weekday_data =      np.array(all_xy[:,528:535])
    time_data =         np.array(all_xy[:,535:536])
    daytype_data =      np.array(all_xy[:,536:539])
    triptime_data =     np.array(all_xy[:,539]    )

    #print(np.concatenate([time_data, daytype_data], axis=1))

    all_data = torch.tensor(all_xy[:,516:539])
    tmp_x = all_data

#    tmp_x = torch.unsqueeze(all_data, 1)
#    tmp_x = tmp_x.repeat(1, 2, 1)
#    tmp_x = all_data.repeat(1, 2)

    tmp_x = tmp_x.to(device)
    print(tmp_x.shape)

#    print(tmp_x)
#    print(tmp_x.shape)
#    print(tmp_x[0].shape)
#    print(tmp_x.shape)

    #tmp_x = all_xy[:,0:539]
    #tmp_y = all_xy[:,539]
#    print(tmp_x)
#    print(tmp_y)

    tmp_y = triptime_data
    self.x = tmp_x.to(device)

#    self.x = torch.tensor(tmp_x, dtype=torch.float32).to(device)
#    self.x = torch.tensor(tmp_x).to(device)

    print(self.x)
    self.y = torch.tensor(tmp_y, dtype=torch.float32).to(device)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, index):
    return self.x[index], self.y[index]

taxi_train = taxiData('jtan-onehot-train.csv')
print(taxi_train)

###################################
batch_size = 200
train_loader = torch.utils.data.DataLoader(taxi_train, batch_size=batch_size, shuffle=True)

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(23, 600)
    self.drop1 = nn.Dropout()
    self.norm1 = nn.BatchNorm1d(600)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(600, 600)
    self.drop2 = nn.Dropout()
    self.norm2 = nn.BatchNorm1d(600)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(600, 600)
    self.drop3 = nn.Dropout()
    self.norm3 = nn.BatchNorm1d(600)
    self.relu3 = nn.ReLU()
    self.fc4 = nn.Linear(600, 600)
    self.drop4 = nn.Dropout()
    self.norm4 = nn.BatchNorm1d(600)
    self.relu4 = nn.ReLU()
    self.fc5 = nn.Linear(600, 600)
    self.drop5 = nn.Dropout()
    self.norm5 = nn.BatchNorm1d(600)
    self.relu5 = nn.ReLU()
    self.fc6 = nn.Linear(600, 600)
    self.drop6 = nn.Dropout()
    self.norm6 = nn.BatchNorm1d(600)
    self.relu6 = nn.ReLU()
    self.fc7 = nn.Linear(600, 1)
  def forward(self, x):
    x = self.drop1(self.relu1(self.norm1(self.fc1(x))))
    x = self.drop2(self.relu2(self.norm2(self.fc2(x))))
    x = self.drop3(self.relu3(self.norm3(self.fc3(x))))
    x = self.drop4(self.relu4(self.norm4(self.fc4(x))))
    x = self.drop5(self.relu5(self.norm5(self.fc5(x))))
    x = self.drop6(self.relu6(self.norm6(self.fc6(x))))
    x = self.fc7(x)
    return x
    

model = MLP()

model.to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.00000001)

#print(train_loader)

epochs = 300
for epoch in range(epochs):
  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
#    print(i, data)
    params, trip_time = data[0].to(device), data[1].to(device)
#    params, trip_time = params.float(), trip_time.float()
    trip_time = trip_time.reshape((trip_time.shape[0], 1))
    '''
    print(params)
    print(params.shape)
    print(trip_time)
    print(trip_time.shape)
    '''

    optimizer.zero_grad()

    outputs = model(params)
#    print(outputs)
    loss = loss_fn(outputs, trip_time)
    loss.backward()
    model.float()
    optimizer.step()
    running_loss += loss.item()

#    if i % 2000 == 1999:    # print every 2000 mini-batches
#       print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.7f}')
  print(f'[{epoch + 1}] loss: {running_loss / i:.13f}')
  running_loss = 0.0

PATH = './save.pth'
torch.save(model.state_dict(), PATH)

print("======= DONE =======")
