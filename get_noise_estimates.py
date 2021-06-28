import numpy as np
from scipy import io
import inference_preprocess
from scipy.signal import butter, filtfilt, find_peaks
from scipy.sparse import eye, spdiags
import matplotlib.pyplot as plt
import h5py
import cv2
import torch
p

def get_noise_estimates():
    mask_path = "/home/js/Desktop/Data/Pytorch_rppgs_save/preprocessing/mask/COHFACE_mask_train.hdf5"
    # Data Load - mask.mat file
    Attention_mask_file = h5py.File(mask_path, 'r')
    Attention_mask1 = Attention_mask_file['Attention_mask1']
    Attention_mask1 = np.array(Attention_mask1[:100000])
    # Attention_mask2 = Attention_mask_file['Attention_mask2']
    # Attention_mask2 = Attention_mask2[:10000]
    Appearance = Attention_mask_file['Appearance']
    Appearance = np.array(Appearance[:100000])
    Pulse_estimate = Attention_mask_file['Pulse_estimate']
    Pulse_estimate = np.transpose(np.array(Pulse_estimate)).reshape(-1,1)
    Pulse_estimate = Pulse_estimate[:100000]
    print('Data Load')
    print(len(Attention_mask1))
    # Inverse Attention Mask Parameter
    mask_noise = np.zeros_like(Attention_mask1)
    L = 36
    fs = 30
    [b_pulse, a_pulse] = butter(9, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    print('butter worth Filter')
    Noise_estimate = h5py.File('/home/js/Desktop/Data/Pytorch_rppgs_save/preprocessing/mask/COHFACE_mask_train_Noise.hdf5',
                               'w')

    # Normalize
    for i in range(len(Attention_mask1)):
        # cv2.imshow("Attention_mask", Attention_mask1[i])
        mask_tmp = Attention_mask1[i] - np.min(Attention_mask1[i])
        mask_tmp = mask_tmp / np.max(Attention_mask1[i])

        mask_noise_tmp = mask_tmp
        threshold = 0.1

        mask_noise_tmp = np.where(mask_noise_tmp > threshold, 0, 1.0)

        mask_noise[i] = mask_noise_tmp
        # cv2.imshow("Inverse Mask", mask_noise[i])

    mask_noise = np.squeeze(mask_noise)
    print('End Normalize')
    GT_stop = 1
    for Channel in range(0, 3):
        mask_noise_i_tmp = mask_noise
        mask_noise_channel = np.zeros((len(Pulse_estimate), L, L))

        for frame in range(len(Appearance)):
            # dXsub_resize = cv2.resize(Appearance[frame], dsize=(L, L), interpolation=cv2.INTER_CUBIC)
            dXsub_resize = Appearance[frame]
            dXsub_resize = np.where(dXsub_resize > 1, 1, dXsub_resize)
            dXsub_resize = np.where(dXsub_resize < 1 / 255, 1 / 255, dXsub_resize)

            # Multiply
            mask_noise_channel[frame, :, :] = np.squeeze(mask_noise_i_tmp[frame]) * dXsub_resize[:, :, Channel]

        mask_noise_channel_mean = np.mean(np.mean(mask_noise_channel, axis=2), axis=1)
        yptest_sub1 = np.cumsum(Pulse_estimate)
        print('End Multiply')

        if GT_stop == 1:
            yptest_sub1_detrend = inference_preprocess.detrend(yptest_sub1, 100)
            yptest_sub2 = filtfilt(b_pulse, a_pulse, np.double(yptest_sub1_detrend))
            print('End Detrend')

        # plt.rcParams["figure.figsize"] = (14, 5)
        # plt.plot(range(len(yptest_sub1_detrend[0:300])), yptest_sub1_detrend[0:300], label='Pulse Estimate')
        # plt.plot(range(len(yptest_sub2[0:300])), yptest_sub2[0:300], label='filter')
        # plt.legend(fontsize='x-large')
        # plt.show()

        mask_noise_channel_mean_detrend = inference_preprocess.detrend(mask_noise_channel_mean, 100)
        mask_noise_channel_mean2 = filtfilt(b_pulse, a_pulse, np.double(mask_noise_channel_mean_detrend))

        # plt.rcParams["figure.figsize"] = (14, 5)
        # plt.plot(range(len(mask_noise_channel_mean_detrend[0:300])), mask_noise_channel_mean_detrend[0:300],
        #          label='Noise Estimate'+str(Channel))
        # plt.plot(range(len(mask_noise_channel_mean2[0:300])),
        #          mask_noise_channel_mean2[0:300], label='filter')
        # plt.legend(fontsize='x-large')
        # plt.show()

        # rename variables to red, green, channel:
        if Channel == 0:
            mask_noise_red_mean2 = mask_noise_channel_mean2
            Noise_estimate.create_dataset('red', data=mask_noise_red_mean2)
            Noise_estimate.create_dataset('Pulse_estimate', data=yptest_sub2)
            GT_stop = 0
            print("red!")

        elif Channel == 1:
            mask_noise_green_mean2 = mask_noise_channel_mean2
            Noise_estimate.create_dataset('green', data=mask_noise_green_mean2)
            print("green!")

        elif Channel == 2:
            mask_noise_blue_mean2 = mask_noise_channel_mean2
            Noise_estimate.create_dataset('blue', data=mask_noise_blue_mean2)
            Noise_estimate.close()
            print('blue!')

get_noise_estimates()