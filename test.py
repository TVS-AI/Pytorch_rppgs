import h5py
import scipy.signal
import torch
import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import butter
from torch.utils.tensorboard import SummaryWriter

from inference_preprocess import detrend


def test(model, test_loader, check_model, device):
    writer = SummaryWriter()
    model = model.to(device)
    checkpoint = torch.load(check_model + "/checkpoint_9d_9h_48m.pth")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for i in range(0, 1):
        with torch.no_grad():
            val_output = torch.zeros(len(test_loader))
            target = torch.zeros(len(test_loader))
            Attention_mask1 = torch.zeros((len(test_loader), 1, 36, 36))
            Attention_mask2 = torch.zeros((len(test_loader), 1, 18, 18))
            Appearance = torch.zeros((len(test_loader), 3, 36, 36))
            for k, (avg, mot, lab) in enumerate(test_loader):
                avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)
                output = model(avg, mot)
                val_output[k] = output[0]
                target[k] = lab[0][0]
                Attention_mask1[k] = output[1]
                Attention_mask2[k] = output[2]
                Appearance[k] = avg
            writer.close()
            print("End : Inference")

            # Save Attention Mask
            data_file = h5py.File('/home/js/Desktop/Data/Pytorch_rppgs_save/preprocessing/mask/UBFC_mask_49.hdf5', 'w')
            data_file.create_dataset('Attention_mask1', data=Attention_mask1.permute(0, 2, 3, 1).numpy())
            data_file.create_dataset('Attention_mask2', data=Attention_mask2.permute(0, 2, 3, 1).numpy())
            data_file.create_dataset('Appearance', data=Appearance.permute(0, 2, 3, 1).numpy())
            data_file.create_dataset('Pulse_estimate', data=val_output)
            data_file.create_dataset('Ground_Truth', data=target)
            data_file.close()
            print("End : Save Mask")

    fs = 30
    low = 0.75 / (0.5 * fs)
    high = 2.5 / (0.5 * fs)
    pulse_pred = detrend(np.cumsum(val_output), 100)
    [b_pulse, a_pulse] = butter(1, [low, high], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
    pulse_pred = pulse_pred / max(abs(pulse_pred))
    target = target / max(abs(target))

    # matplot---------------------------------------------------------------------
    plt.rcParams["figure.figsize"] = (14, 5)
    plt.plot(range(len(pulse_pred[0:300])), pulse_pred[0:300], label='inference')
    plt.plot(range(len(target[0:300])), target[0:300], label='target')
    plt.legend(fontsize='x-large')
    plt.show()
