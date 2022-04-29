import multiprocessing
import os

import h5py
import numpy as np
from utils.image_preprocess import Deepphys_preprocess_Video, PhysNet_preprocess_Video, RTNet_preprocess_Video, \
    GCN_preprocess_Video, Axis_preprocess_Video
from utils.seq_preprocess import PPNet_preprocess_Mat
from utils.text_preprocess import Deepphys_preprocess_Label, PhysNet_preprocess_Label, GCN_preprocess_Label, \
    Axis_preprocess_Label
from utils.funcs import store_as_list_of_dicts, load_list_of_dicts
from scipy import io


def preprocessing(save_root_path: str = "/media/hdd1/dy_dataset/",
                  model_name: str = "DeepPhys",
                  data_root_path: str = "/media/hdd1/",
                  dataset_name: str = "UBFC",
                  train_ratio: float = 0.8):
    """
    :param save_root_path: save file destination path
    :param model_name: select preprocessing method
    :param data_root_path: data set root path
    :param dataset_name: data set name(ex. UBFC, COFACE)
    :param train_ratio: data split [ train ratio : 1 - train ratio]
    :return:
    """
    split_flag = False
    dataset_root_path = data_root_path + dataset_name

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    if dataset_name == "UBFC":
        data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("subject")]
        vid_name = "/vid.avi"
        ground_truth_name = "/ground_truth.txt"
    elif dataset_name == "cuff_less_blood_pressure":
        data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("part")]
    elif dataset_name == "VIPL_HR":
        data_dir = "/data"
        person_data_path = dataset_root_path + data_dir
        # source1/2/3
        # /data/pxx/vxx/sourcex/
        data_list = []
        person_list = [data for data in os.listdir(person_data_path) if data.__contains__("p")]
        for person in person_list:
            v_list = [v for v in os.listdir(person_data_path + "/" + person) if v.__contains__(v)]
            for v in v_list:
                source_list = [source for source in os.listdir(person_data_path + "/" + person + "/" + v) if
                               not source.__contains__("4")]
                for source in source_list:
                    data_list.append(data_dir + "/" + person + "/" + v + "/" + source)

        vid_name = "/video.avi"
        ground_truth_name = "/wave.csv"

        print(person_list)
    elif dataset_name.__contains__("cohface"):
        dataset_root_path = data_root_path + "cohface"
        protocol = dataset_root_path + "/" + "protocols/"
        if dataset_name.__contains__("all"):
            protocol += "all/all.txt"
        elif dataset_name.__contains__("clean"):
            protocol += "clean/all.txt"
        elif dataset_name.__contains__("natural"):
            protocol += "natural/all.txt"
        f = open(protocol,'r')
        data_list = f.readlines()
        data_list = [path.replace("data\n","") for path in data_list]
        f.close()
        vid_name = "data.mkv"
        ground_truth_name = "data.hdf5"
    elif dataset_name.__contains__("PURE"):
        data_list = os.listdir(dataset_root_path)
        vid_name = "/png"
        ground_truth_name = "/json"

    process = []

    # multiprocessing
    if split_flag == False:
        for index, data_path in enumerate(data_list):
            proc = multiprocessing.Process(target=preprocess_Dataset,
                                           args=(dataset_root_path + "/" + data_path, vid_name, ground_truth_name, 2,
                                                 model_name, return_dict))
            # (path, vid_name, ground_truth_name, flag, model_name, return_dict):
            # flag 0 : pass
            # flag 1 : detect face
            # flag 2 : remove nose
            process.append(proc)
            proc.start()

        for proc in process:
            proc.join()
    else:
        loop = len(data_list)//32
        loop = 5

        for i in range(loop):
            for index, data_path in enumerate(data_list[i*32:(i+1)*32]):
                proc = multiprocessing.Process(target=preprocess_Dataset,
                                               args=(dataset_root_path + "/" + data_path, vid_name, ground_truth_name, 2, model_name, return_dict))
                # flag 0 : pass
                # flag 1 : detect face
                # flag 2 : remove nose
                process.append(proc)
                proc.start()

            for proc in process:
                proc.join()

    train = int(len(return_dict.keys()) * train_ratio)  # split dataset
    dt = h5py.special_dtype(vlen=np.float32)
    train_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_train.hdf5", "w")

    if model_name in ["DeepPhys", "PhysNet", "PhysNet_LSTM"]:

        for index, data_path in enumerate(return_dict.keys()[:train]):
            dset = train_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        train_file.close()

        test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
        for index, data_path in enumerate(return_dict.keys()[train:]):
            dset = test_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        test_file.close()

    elif model_name in ["PPNet"]:

        for index, data_path in enumerate(return_dict.keys()[:train]):
            dset = train_file.create_group(data_path)
            dset['ppg'] = return_dict[data_path]['ppg']
            dset['sbp'] = return_dict[data_path]['sbp']
            dset['dbp'] = return_dict[data_path]['dbp']
            dset['hr'] = return_dict[data_path]['hr']
        train_file.close()

        test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
        for index, data_path in enumerate(return_dict.keys()[train:]):
            dset = test_file.create_group(data_path)
            dset['ppg'] = return_dict[data_path]['ppg']
            dset['sbp'] = return_dict[data_path]['sbp']
            dset['dbp'] = return_dict[data_path]['dbp']
            dset['hr'] = return_dict[data_path]['hr']
        test_file.close()
    elif model_name in ["GCN","GCN_TEST"]:
        for index, data_path in enumerate(return_dict.keys()[:train]):
            dset = train_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        train_file.close()

        test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
        for index, data_path in enumerate(return_dict.keys()[train:]):
            dset = test_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        test_file.close()
    elif model_name in ["AxisNet"]:
        for index, data_path in enumerate(return_dict.keys()[:train]):
            dset = train_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
            dset['preprocessed_ptt'] = return_dict[data_path]['preprocessed_ptt']
        train_file.close()

        test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
        for index, data_path in enumerate(return_dict.keys()[train:]):
            dset = test_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
            dset['preprocessed_ptt'] = return_dict[data_path]['preprocessed_ptt']
        test_file.close()

        # train_graph_file = save_root_path + model_name + "_"+dataset_name + "_train.pkl"
        # test_graph_file = save_root_path + model_name + "_"+dataset_name + "_test.pkl"
        #
        # saved_graph = []
        # for index, data_path in enumerate(return_dict.keys()[:train]):
        #     dset = train_file.create_group(data_path)
        #     dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
        #     dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        #     saved_graph.extend(return_dict[data_path]['preprocessed_graph'])
        # train_file.close()
        #
        # store_as_list_of_dicts(train_graph_file,saved_graph)
        #
        # saved_graph = []
        # test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
        # for index, data_path in enumerate(return_dict.keys()[train:]):
        #     dset = test_file.create_group(data_path)
        #     dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
        #     dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        #     saved_graph.extend(return_dict[data_path]['preprocessed_graph'])
        # test_file.close()
        #
        # store_as_list_of_dicts(test_graph_file, saved_graph)


def preprocess_Dataset(path, vid_name, ground_truth_name, flag, model_name, return_dict):
    """
    :param path: dataset path
    :param flag: face detect flag
    :param model_name: select preprocessing method
    :param return_dict: : preprocessed image, label
    """

    # Video Based
    if model_name == "DeepPhys":
        rst, preprocessed_video = Deepphys_preprocess_Video(path + vid_name, flag)
    elif model_name == "PhysNet" or model_name == "PhysNet_LSTM":
        rst, preprocessed_video,frame_total = PhysNet_preprocess_Video(path + vid_name, flag)
    elif model_name == "RTNet":
        rst, preprocessed_video = RTNet_preprocess_Video(path + vid_name, flag)
    elif model_name == "PPNet":  # Sequence data based
        ppg, sbp, dbp, hr = PPNet_preprocess_Mat(path)
    elif model_name == "GCN":
        # rst, preprocessed_video, saved_graph = GCN_preprocess_Video(path + "/vid.avi", flag)
        rst, preprocessed_video, sliding_window_stride = GCN_preprocess_Video(path + vid_name, flag)
    elif model_name == "AxisNet":
        rst, preprocessed_video, sliding_window_stride, num_frames, stacked_ptts = Axis_preprocess_Video(
            path + vid_name, flag)
    # rst,bvp,sliding,frames,ptt
    if model_name in ["DeepPhys", "MTTS", "PhysNet", "PhysNet_LSTM"]:  # can't detect face
        if not rst:
            return

    if model_name == "DeepPhys":
        preprocessed_label = Deepphys_preprocess_Label(path + ground_truth_name)
    elif model_name == "PhysNet" or model_name == "PhysNet_LSTM":
        preprocessed_label,start,end = PhysNet_preprocess_Label(path + ground_truth_name, frame_total)
        # preprocessed_label = PhysNet_preprocess_Label(path + ground_truth_name, frame_total)
    elif model_name == "GCN":
        preprocessed_label = GCN_preprocess_Label(path + ground_truth_name, sliding_window_stride)
    elif model_name == "AxisNet":
        preprocessed_label = Axis_preprocess_Label(path + ground_truth_name, sliding_window_stride, num_frames)





    # ppg, sbp, dbp, hr
    if model_name in ["DeepPhys", "PhysNet", "PhysNet_LSTM"]:
        return_dict[path.replace('/','')] = {'preprocessed_video': preprocessed_video[start:end],
                                            'preprocessed_label': preprocessed_label[start:end]}
    elif model_name in ["PPNet"]:
        return_dict[path.replace('/','')] = {'ppg': ppg, 'sbp': sbp, 'dbp': dbp, 'hr': hr}
    elif model_name in ["GCN"]:
        return_dict[path.replace('/','')] = {'preprocessed_video': preprocessed_video,
                                            'preprocessed_label': preprocessed_label}
    elif model_name in ["AxisNet"]:
        return_dict[path.replace('/', '')] = {'preprocessed_video': preprocessed_video,
                                            'preprocessed_ptt': stacked_ptts,
                                            'preprocessed_label': preprocessed_label}
        # 'preprocessed_graph': saved_graph}
