import h5py
import torch
from model import Denoising_LSTM
import torch.optim as optim
from torch.utils.data import DataLoader
from LSTM_dataset import timeseries
import datetime
from torchsummary import summary

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
train_data_set = h5py.File('/home/js/Desktop/Data/Pytorch_rppgs_save/preprocessing/noise/UBFC_noise_Data.hdf5', 'r')
Xtrain = train_data_set['train'][:]
Ytrain = train_data_set['GT'][:]
LSTM = Denoising_LSTM()
summary(LSTM, (60, 4))
LSTM.to(device)
MSEloss = torch.nn.MSELoss()
Adam = optim.Adam(LSTM.parameters(), lr=0.001)
n_epoch = 11

dataset = timeseries(Xtrain, Ytrain)
train_loader = DataLoader(dataset, batch_size=32)

for epoch in range(0, n_epoch):
    running_loss = 0.0
    for batch_idx, (input_data, GT) in enumerate(train_loader):
        Adam.zero_grad()
        input_data, GT = input_data.to(device), GT.to(device)
        output = LSTM(input_data)
        loss = MSEloss(output,GT)
        loss.backward()
        Adam.step()
        running_loss += loss.item()
        print('Train : [%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 32))
