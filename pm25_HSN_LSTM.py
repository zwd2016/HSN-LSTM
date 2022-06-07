# -*- coding: utf-8 -*-
"""

@author: Wendong
"""

import torch
from torch import nn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from network_slstm import HSN_LSTM
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import time
from math import sqrt
np.random.seed(1337)  # for reproducibility
np.set_printoptions(threshold=np.inf)
import matplotlib
matplotlib.use('Agg')


# logger txt: python console to x.txt
class Logger(object):
    def __init__(self, filename='./pm25_HSN_LSTM/default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
time_log_begin = time.ctime()
filename_log =  './pm25_HSN_LSTM/pm25_HSN_LSTM_'+str(time_log_begin)+'.log' 
filename_log1 = './pm25_HSN_LSTM/pm25_HSN_LSTM_error_warning_'+str(time_log_begin)+'.log'
sys.stdout = Logger(filename_log, sys.stdout)
sys.stderr = Logger(filename_log1, sys.stderr)


data = pd.read_csv("./PM25/PRSA_data_2010.1.1-2014.12.31.csv")
cbwd = pd.get_dummies(data['cbwd'])
cbwd.columns = ["cbwd_{}".format(i) for i in range(cbwd.shape[1])]
data = pd.concat([data, cbwd], axis=1)
data.drop(['cbwd'], axis=1, inplace=True)
print('pm25_data_head:',data.head())
cols = list(data.columns[5:])
print('cols name:',cols)
data['pm2.5'] = data['pm2.5'].fillna(method='ffill').fillna(method='bfill')
depth = 10#timesteps or time windows

X = np.zeros((len(data), depth, len(cols)))
for i, name in enumerate(cols):
    for j in range(depth):
        X[:, j, i] = data[name].shift(depth - j - 1).fillna(method='bfill')
prediction_horizon = 6 #multi-horizons
y = data['pm2.5'].shift(-prediction_horizon).fillna(method='ffill').values

train_bound = int(0.6*(len(data)))
val_bound = int(0.8*(len(data)))

X_train = X[:train_bound]
X_val = X[train_bound:val_bound]
X_test = X[val_bound:]
y_train = y[:train_bound]
y_val = y[train_bound:val_bound]
y_test = y[val_bound:]

X_train_min, X_train_max = X_train.min(axis=0), X_train.max(axis=0)
y_train_min, y_train_max = y_train.min(axis=0), y_train.max(axis=0)

X_train = (X_train - X_train_min)/(X_train_max - X_train_min + 1e-9)
X_val = (X_val - X_train_min)/(X_train_max - X_train_min + 1e-9)
X_test = (X_test - X_train_min)/(X_train_max - X_train_min + 1e-9)
y_train = (y_train - y_train_min)/(y_train_max - y_train_min +1e-9)
y_val = (y_val - y_train_min)/(y_train_max - y_train_min + 1e-9)
y_test = (y_test - y_train_min)/(y_train_max - y_train_min + 1e-9)

X_train_t = torch.Tensor(X_train)
X_val_t = torch.Tensor(X_val)
X_test_t = torch.Tensor(X_test)
y_train_t = torch.Tensor(y_train)
y_val_t = torch.Tensor(y_val)
y_test_t = torch.Tensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)


model = HSN_LSTM(X_train_t.shape[2], 1, 128).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 12, gamma=0.92)


epochs = 1000
loss = nn.MSELoss()
patience = 150
min_val_loss = 9999
counter = 0
total_time_start = time.time()
for i in range(epochs):
    epochs_time_start = time.time()
    
    mse_train = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        opt.zero_grad()
        y_pred = model(batch_x)
        y_pred = y_pred.squeeze(1)
        l = loss(y_pred, batch_y)
        l.backward()
        mse_train += l.item()*batch_x.shape[0]
        opt.step()
    epoch_scheduler.step()
    with torch.no_grad():
        mse_val = 0
        preds = []
        true = []
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output = model(batch_x)
            output = output.squeeze(1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            mse_val += loss(output, batch_y).item()*batch_x.shape[0]
    preds = np.concatenate(preds)
    true = np.concatenate(true)
    
    epochs_time_end = time.time()
    
    if min_val_loss > mse_val**0.5:
        min_val_loss = mse_val**0.5
        print("Saving...")
        torch.save(model.state_dict(), "./pm25_HSN_LSTM/HSN_LSTM_pm25.pt")
        counter = 0
    else: 
        counter += 1
    
    if counter == patience:
        break
    print("Iter: ", i, "train loss: ", (mse_train/len(X_train_t))**0.5, "val loss: ", (mse_val/len(X_val_t))**0.5, "running time (train+val_SLSTM)/epochs (seconds): ", epochs_time_end-epochs_time_start)
    
    #training loss value write to .txt 
    tl = (mse_train/len(X_train_t))**0.5
    file_1=open('./pm25_HSN_LSTM/train_loss.txt','a')
    file_1.write(str(tl)+",");
    file_1.close()
    #val loss value write to .txt
    vl = (mse_val/len(X_val_t))**0.5
    file_2=open('./pm25_HSN_LSTM/val_loss.txt','a')
    file_2.write(str(vl)+",");# , represents the split symbol
    file_2.close()
    
    if(i % 10 == 0):
        preds = preds*(y_train_max - y_train_min) + y_train_min
        true = true*(y_train_max - y_train_min) + y_train_min
        rmse = sqrt(mean_squared_error(true, preds))
        mae = mean_absolute_error(true, preds)
        print("lr: ", opt.param_groups[0]["lr"])
        print("test mse: ", rmse, "test mae: ", mae)
        plt.figure(figsize=(20, 10))
        plt.tick_params(labelsize=14)
        plt.title('PM2.5 dataset (on the test set): No_'+ str(i) + '_Epochs',fontsize=12)
        plt.xlabel('Time range (/1 hour)',fontsize=14)
        plt.ylabel('PM2.5 air quality index range',fontsize=14)
        plt.grid()
        plt.plot(true,label='true')
        plt.plot(preds,label='predictions HSN_LSTM')
        plt.legend(fontsize=14)
        filename = './pm25_HSN_LSTM/' + 'Figure_pred_true_Epochs_' + str(i) + '_HSN_LSTM' + '.pdf'
        plt.savefig(filename, dpi=1200)
        plt.close()

total_time_end = time.time()
print('Total running time of the pm25_HSN_LSTM method :',total_time_end-total_time_start)
model.load_state_dict(torch.load("./pm25_HSN_LSTM/HSN_LSTM_pm25.pt"))

with torch.no_grad():
    mse_val = 0
    preds = []
    true = []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        output = model(batch_x)
        output = output.squeeze(1)
        preds.append(output.detach().cpu().numpy())
        true.append(batch_y.detach().cpu().numpy())
        mse_val += loss(output, batch_y).item()*batch_x.shape[0]
preds = np.concatenate(preds)
true = np.concatenate(true)

mse_regular = mean_squared_error(true, preds)
mae_regular = mean_absolute_error(true, preds)
rmse_regular = sqrt(mean_squared_error(true, preds))

print('HSN_LSTM test regularization mse:', mse_regular)
print('HSN_LSTM test regularization rmse:', rmse_regular)
print('HSN_LSTM test regularization mae:', mae_regular)

preds = preds*(y_train_max - y_train_min) + y_train_min
true = true*(y_train_max - y_train_min) + y_train_min
#pred-true value write to .txt
a = true
file1=open('./pm25_HSN_LSTM/y_true.txt','w')
file1.write(str(a));
file1.close()

b = preds
file2=open('./pm25_HSN_LSTM/pred_yhat_HSN_LSTM.txt','w')
file2.write(str(b));
file2.close()

mse = mean_squared_error(true, preds)
mae = mean_absolute_error(true, preds)
rmse = sqrt(mean_squared_error(true, preds))

print('HSN_LSTM test mse:', mse)
print('HSN_LSTM test rmse:', rmse)
print('HSN_LSTM test mae:', mae)

plt.figure(figsize=(20, 10))
plt.tick_params(labelsize=14)
plt.title('PM2.5 dataset (on the test set)',fontsize=14)
plt.xlabel('Time range (/1 hour)',fontsize=14)
plt.ylabel('PM2.5 air quality index range',fontsize=14)
plt.grid()
plt.plot(true,label='Ground Truth')
plt.plot(preds,label='Prediction HSN_LSTM')
filename = './pm25_HSN_LSTM/' + 'Figure_pred_true_HSN_LSTM' + '.pdf'
plt.legend(fontsize=14)
plt.savefig(filename, dpi=1200)
plt.close()

