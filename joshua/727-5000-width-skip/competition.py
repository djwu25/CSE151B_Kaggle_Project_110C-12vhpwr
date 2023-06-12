#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

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
    print('LOADED')
    all_xy = df.to_numpy()
    
#    tmp_x = all_xy[:,0:7]
 
    call_type_data =    np.array(all_xy[:,0:66]     )
    taxi_id_data =      np.array(all_xy[:,66:516]   )
    month_data =        np.array(all_xy[:,516:528]  )
    weekday_data =      np.array(all_xy[:,528:535]  )
    time_data =         np.array(all_xy[:,535:536]  , dtype=np.float32)
    daytype_data =      np.array(all_xy[:,536:539]  )
    triptime_data =     np.array(all_xy[:,539]      , dtype=np.float32)

    #print(np.concatenate([time_data, daytype_data], axis=1))

    all_data = torch.tensor(all_xy[:,0:539], dtype=torch.float32)
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

taxi_train = taxiData('jtan-onehot-train-clean.csv')
print(taxi_train)

###################################
batch_size = 512
train_loader = torch.utils.data.DataLoader(taxi_train, batch_size=batch_size, shuffle=True)

thicc = 5000
print(thicc)

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

#PATH = './epoch5-save.pth'
#model.load_state_dict(torch.load(PATH))

model.to(device)
print(model)

# loss_fn = nn.MSELoss()

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

loss_fn = RMSELoss

optimizer = optim.SGD(model.parameters(), lr=0.0001)

#print(train_loader)

epochs = 250
for epoch in tqdm(range(epochs)):
  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
    params, trip_time = data[0].to(device), data[1].to(device)
    trip_time = trip_time.float()
    trip_time = trip_time.reshape((trip_time.shape[0], 1))

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

  PATH = f'./epoch{epoch}-save.pth'
  torch.save(model.state_dict(), PATH)

PATH = f'./save.pth'
torch.save(model.state_dict(), PATH)

print("======= DONE =======")
