import numpy as np
import matplotlib.pyplot as plt
import h5py


# Load Mask noise
noise = h5py.File("/home/js/Desktop/Data/Pytorch_rppgs_save/preprocessing/mask/UBFC_mask_49_Noise.hdf5", 'r')

noise_red = noise['red'][:]
noise_blue = noise['blue'][:]
noise_green = noise['green'][:]
noise_pulse = noise['Pulse_estimate'][:]

noise_red = noise_red / max(abs(noise_red))
noise_blue = noise_blue / max(abs(noise_blue))
noise_green = noise_green / max(abs(noise_green))
noise_pulse = noise_pulse / max(abs(noise_pulse))

noise.close()

# Load GT
ground_truth = h5py.File('/home/js/Desktop/Data/Pytorch_rppgs_save/preprocessing/mask/UBFC_mask_test_49.hdf5', 'r')
GT = ground_truth['Ground_Truth']

# plt
plt.rcParams["figure.figsize"] = (14, 5)
plt.plot(range(len(noise_red[0:300])), noise_red[0:300], label='noise_red')
plt.plot(range(len(noise_blue[0:300])), noise_blue[0:300], label='noise_blue')
plt.plot(range(len(noise_green[0:300])), noise_green[0:300], label='noise_green')
plt.plot(range(len(noise_pulse[0:300])), noise_pulse[0:300], label='noise_pulse')
plt.legend(fontsize='x-large')
plt.show()

n_number = 4  # noise_red, noise_blue, noise_green, noise_pulse_estimate
n_example = 60  # length of the signals input to the LSTM

signal_length = len(noise_red)
n_sample = len(range(0, signal_length - (n_example - 1), int(n_example / 2)))

# train_data
train_red = np.zeros((n_sample, n_example), dtype=np.float32)
train_blue = np.zeros((n_sample, n_example), dtype=np.float32)
train_green = np.zeros((n_sample, n_example), dtype=np.float32)
train_pulse = np.zeros((n_sample, n_example), dtype=np.float32)
train_data = np.zeros((n_sample, n_example, 4), dtype=np.float32)
train_gt = np.zeros((n_sample, n_example), dtype=np.float32)

# test_data
test_red = np.zeros((n_sample, n_example), dtype=np.float32)
test_blue = np.zeros((n_sample, n_example), dtype=np.float32)
test_green = np.zeros((n_sample, n_example), dtype=np.float32)
test_pulse = np.zeros((n_sample, n_example), dtype=np.float32)
test_data = np.zeros((n_sample, n_example, 4), dtype=np.float32)
test_gt = np.zeros((n_sample, n_example), dtype=np.float32)

# overlap Data
count = 0
for i in range(0, signal_length - (n_example - 1), int(n_example / 2)):
    train_red[count, 0:n_example] = noise_red[i: i + n_example]
    train_blue[count, 0:n_example] = noise_blue[i: i + n_example]
    train_green[count, 0:n_example] = noise_green[i: i + n_example]
    train_pulse[count, 0:n_example] = noise_pulse[i: i + n_example]
    train_gt[count, 0:n_example] = GT[i: i + n_example]
    count += 1

train_data[:, :, 0] = train_red
train_data[:, :, 1] = train_blue
train_data[:, :, 2] = train_green
train_data[:, :, 3] = train_pulse
train_gt = train_gt.reshape(train_gt.shape[0], train_gt.shape[1], 1)

train_data_set = h5py.File('/home/js/Desktop/Data/Pytorch_rppgs_save/preprocessing/noise/UBFC_noise_test_49.hdf5', 'w')
train_data_set.create_dataset('train', data=train_data)
train_data_set.create_dataset('GT', data=train_gt)
train_data_set.close()

print('test')
