import h5py
import scipy.signal
import torch
import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import butter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from LSTM_dataset import timeseries
from inference_preprocess import detrend
from model import Denoising_LSTM

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
test_dataset = h5py.File('/home/js/Desktop/Data/Pytorch_rppgs_save/preprocessing/noise/UBFC_noise_test_49.hdf5', 'r')
Xtest = test_dataset['train'][:]
Ytest = test_dataset['GT'][:]
dataset = timeseries(Xtest, Ytest)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
writer = SummaryWriter()
LSTM = Denoising_LSTM().to(device)
checkpoint = torch.load("/home/js/Desktop/Data/Pytorch_rppgs_save/model_checkpoint/checkpoint_lstm14d_11h_57m.pth")
LSTM.load_state_dict(checkpoint['state_dict'])
LSTM.eval()
for i in range(0, 1):
    val_output = torch.zeros((len(test_loader), 60))
    target = torch.zeros((len(test_loader), 60))
    val_output_list = []
    target_list = []
    with torch.no_grad():
        for batch_idx, (val_data, val_GT) in enumerate(test_loader):
            val_data, val_GT = val_data.to(device), val_GT.to(device)
            output = LSTM(val_data)
            val_output[batch_idx] = output.squeeze()
            val_output_list = val_output_list + output.squeeze().tolist()
            target[batch_idx] = val_GT.squeeze()
            target_list = target_list + val_GT.squeeze().tolist()
        writer.close()
        print("End : Inference")

# fs = 30
# low = 0.75 / (0.5 * fs)
# high = 2.5 / (0.5 * fs)
# pulse_pred = detrend(np.cumsum(val_output), 100)
# [b_pulse, a_pulse] = butter(1, [low, high], btype='bandpass')
# pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
val_output = val_output.flatten()
target = target.flatten()
val_output = val_output / max(abs(val_output))
target = target / max(abs(target))
corr = np.corrcoef(val_output, target)
print(corr)
k = 0
# matplot---------------------------------------------------------------------
plt.rcParams["figure.figsize"] = (14, 5)
plt.plot(range(len(val_output[k:k+300])), val_output[k:k+300], label='inference')
plt.plot(range(len(target[k:k+300])), target[k:k+300], label='target')
plt.legend(fontsize='x-large')
plt.show()
