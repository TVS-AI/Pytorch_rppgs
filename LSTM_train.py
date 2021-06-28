import h5py
import torch
from model import Denoising_LSTM
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from LSTM_dataset import timeseries
import datetime
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()
train_data_set = h5py.File('/home/js/Desktop/Data/Pytorch_rppgs_save/preprocessing/noise/UBFC_noise_Data.hdf5', 'r')
Xtrain = train_data_set['train'][:]
Ytrain = train_data_set['GT'][:]
LSTM = Denoising_LSTM()
summary(LSTM, (60, 4))
LSTM.to(device)
MSEloss = torch.nn.MSELoss()
Adam = optim.Adam(LSTM.parameters(), lr=0.001)
n_epoch = 1100

x_train, x_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size=0.2)
dataset = timeseries(x_train, y_train)
valset = timeseries(x_test, y_test)
train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(valset, batch_size=32, shuffle=False)
folder_name = datetime.datetime.now()
tmp_val_loss = 100.0

for epoch in range(0, n_epoch):
    running_loss = 0.0
    for batch_idx, (input_data, GT) in enumerate(train_loader):
        Adam.zero_grad()
        input_data, GT = input_data.to(device), GT.to(device)
        output = LSTM(input_data)
        loss = MSEloss(output, GT)
        running_loss += loss.item()
        loss.backward()
        Adam.step()
    print('Train : [%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss))
    writer.add_scalar('train_loss', running_loss, epoch)
    LSTM.eval()
    with torch.no_grad():
        val_loss = 0.0
        for val_idx, (val_data, val_GT) in enumerate(val_loader):
            val_data, val_GT = val_data.to(device), val_GT.to(device)
            val_output = LSTM(val_data)
            val_loss = MSEloss(val_output, val_GT)
            val_loss += val_loss.item()
        print('Validation : [%d, %5d] loss: %.3f' % (epoch + 1, val_idx + 1, val_loss))
        writer.add_scalar('validation_loss', val_loss, epoch)
        if tmp_val_loss > val_loss:
            checkpoint = {'Epoch': epoch,
                          'state_dict': LSTM.state_dict(),
                          'optimizer': Adam.state_dict()}
            torch.save(checkpoint,
                       "/home/js/Desktop/Data/Pytorch_rppgs_save/model_checkpoint" + "/checkpoint_lstm" + str(folder_name.day) + "d_"
                       + str(folder_name.hour) + "h_"
                       + str(folder_name.minute) + 'm.pth')
            tmp_val_loss = val_loss
            print("Update tmp : " + str(tmp_val_loss))
    LSTM.train()
writer.close()

