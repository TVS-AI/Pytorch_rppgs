import numpy as np
from sklearn.preprocessing import minmax_scale
import h5py
from scipy.signal import resample,resample_poly,decimate
import cv2
from utils.funcs import detrend
import csv
from biosppy.signals import bvp
import math
import json
def Deepphys_preprocess_Label(path):
    '''
    :param path: label file path
    :return: delta pulse
    '''
    # TODO : need to check length with video frames
    # TODO : need to implement piecewise cubic Hermite interpolation
    # Load input
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float, label))

    delta_label = []
    for i in range(len(label) - 1):
        delta_label.append(label[i + 1] - label[i])
    delta_label -= np.mean(delta_label)
    delta_label /= np.std(delta_label)
    delta_label = np.array(delta_label).astype('float32')
    delta_pulse = delta_label.copy()  # 이거 왜 있지?
    f.close()

    return delta_pulse


def PhysNet_preprocess_Label(path,frame_total):
    '''
    :param path: label file path
    :return: wave form
    '''
    set = 64
    div = 32
    # Load input
    if path.__contains__("hdf5"):
        f = h5py.File(path,'r')
        label = np.asarray(f['pulse'])
        # label = decimate(label,int(len(label)/frame_total))
        label_bvp = bvp.bvp(label, 256, show=False)
        label = label_bvp['filtered']


        label = smooth(label,128)
        label = resample_poly(label,15,128)
        # label = resample(label,frame_total)
        # label = detrend(label,100)

        start = label_bvp['onsets'][3]
        end = label_bvp['onsets'][-2]
        label = label[start:end]
        # plt.plot(label)
        # label = resample(label,frame_total)
        label -= np.mean(label)
        label /= np.std(label)
        start = math.ceil(start/32)
        end = math.floor(end / 32)

    elif path.__contains__("json"):
        name = path.split("/")
        label = []
        with open(path[:-4]+name[-2]+".json") as json_file:
            json_data = json.load(json_file)
            for data in json_data['/FullPackage']:
                label.append(data['Value']['waveform'])
        label = resample(label,len(label)//2)
    elif path.__contains__("csv"):
        f = open(path,'r')
        rdr = csv.reader(f)
        r = list(rdr)
        label = np.asarray(r[1:]).reshape((-1)).astype(np.float)

        f_time = open(path[:-8]+"time.txt",'r')
        f_time = f_time.read().split('\n')
        time = np.asarray(f_time[:-1]).astype(np.float)
        print("a")

    else:
        f = open(path, 'r')
        f_read = f.read().split('\n')
        label = ' '.join(f_read[0].split()).split()
        label = list(map(float, label))
        label = np.array(label).astype('float32')
    split_raw_label = np.zeros(((len(label) // 32), 32))
    index = 0
    for i in range(len(label) // 32):
        split_raw_label[i] = label[index:index + 32]
        index = index + 32

    return split_raw_label,0,-1

def GCN_preprocess_Label(path,sliding_window_stride):
    '''
    :param path: label file path
    :return: wave form
    '''

    div = 256
    stride = sliding_window_stride
    # Load input
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float, label))
    label = np.array(label).astype('float32')
    num_maps = int((len(label) - div)/stride + 1)
    split_raw_label = np.zeros((num_maps, div))
    index = 0
    for i in range(0,num_maps,stride):
        split_raw_label[i] = label[index:index + div]
        index = index + stride
    f.close()

    return split_raw_label

def Axis_preprocess_Label(path,sliding_window_stride,num_frames,clip_size = 256):
    '''
    :param path: label file path
    :return: wave form
    '''

    # div = 256
    # stride = num_maps
    # Load input
    ext = path.split('.')[-1]


    f = open(path, 'r')

    f_read = f.read().split('\n')
    if ext == 'txt':
        label = ' '.join(f_read[0].split()).split()
        label = list(map(float, label))


    elif ext == 'csv':
        label = f_read[1:]
        label = [float(txt) for txt in label if txt != '']

    label = np.array(label).astype('float32')
    label = np.resize(label,num_frames)
    # print(path + str(len(label))+ "  " + str(num_maps)+"  "+str(clip_size) +"  " + str(sliding_window_stride) + "  " + str(num_frames))
    # print(num_maps)
    num_maps = int((num_frames - clip_size) / sliding_window_stride + 1)
    split_raw_label = np.zeros((num_maps, clip_size))
    index = 0
    for start_frame_index in range(0, num_frames,sliding_window_stride ):
        end_frame_index = start_frame_index + clip_size
        if end_frame_index > num_frames:
            break
        split_raw_label[index,:] = minmax_scale(label[start_frame_index:end_frame_index],axis=0,copy=True)*2-1#label[start_frame_index:end_frame_index]
        index += 1
    f.close()

    return split_raw_label

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth