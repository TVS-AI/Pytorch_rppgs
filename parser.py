import argparse

import torch
import torchsummary
import h5py
import bvpdataset as bp
import numpy as np

from model import model
from preporcessing_faceDetect import DatasetDeepPhysUBFC
from torch.utils.data import DataLoader
from get_noise_estimates import get_noise_estimates
from train import train
from test import test

dir_path = "/home/js/Desktop/Data/Pytorch_rppgs_save"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=0, help="0.CAN 1.MTTS-CAN 2.MetaPhys")
    # preprocessing
    parser.add_argument('--preprocessing', type=int, default=3, help="0.False 1.True")
    # parameter
    parser.add_argument('--GPU_num', type=int, default=1, help="0.Tesla:0 1.Tesla:1")
    parser.add_argument('--loss', type=int, default=0, help="0.MSELoss 1.yet!")
    parser.add_argument('--optimizer', type=int, default=0, help="0.Adadelta 1.yet!")
    parser.add_argument('--mode', type=int, default=1, help="0.Train 1.Test")
    parser.add_argument('--batch_size', type=int, default=32, help="default 32")
    # train mode
    parser.add_argument('--train_data', type=str, default=dir_path + "/preprocessing/train/UBFC_trainset_face.npz")
    parser.add_argument('--epoch', type=int, default=100, help="default 25")
    parser.add_argument('--checkpoint', type=str, default=dir_path + "/model_checkpoint",
                        help=dir_path + "/model_checkpoint")
    # test mode
    parser.add_argument('--test_data', type=str, default=dir_path + "/preprocessing/test/UBFC_train_Data.hdf5")
    parser.add_argument('--check_model', type=str, default=dir_path + "/model_checkpoint")

    # Denoising
    parser.add_argument('--LSTM', type=int, default=1, help="0.False 1.True")
    args = parser.parse_args()

    hyper_params = {
        "model": args.model,
        "preprocessing": args.preprocessing,
        "GPU_num": args.GPU_num,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "train_data": args.train_data,
        "epoch": args.epoch,
        "checkpoint": args.checkpoint,
        "test_data": args.test_data,
        "check_model": args.check_model,
        "LSTM": args.LSTM
    }

    device = torch.device('cuda:' + str(args.GPU_num)) if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    # Load model
    model = model()
    torchsummary.summary(model, ((3, 36, 36), (3, 36, 36)))
    # Loss function
    loss_fn = None
    if args.loss == 0:
        loss_fn = torch.nn.MSELoss()
        print("Select MSELoss")
    elif args.loss == 1:
        print("yet")
        exit(666)
    else:
        print('\nError! No such loss function. Choose from : 0.MSELoss, 1.yet')
        exit(666)

    # Optimizer
    optimizer = None
    if args.optimizer == 0:
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
        print("Select Adadelta")
    elif args.loss == 1:
        print("yet")
        exit(666)
    else:
        print('\nError! No such optimizer. Choose from : 0.Adadelta, 1.yet')
        exit(666)

    # Data Load
    dataset = None
    if args.preprocessing == 0:
        if args.mode == 0:
            dataset = h5py.File(dir_path + '/preprocessing/train/UBFC_train_Data.hdf5', 'r')
            print("Load : " + dir_path + "/preprocessing/train/UBFC_train_Data.hdf5")
        elif args.mode == 1:
            dataset = h5py.File(args.test_data, 'r')
            print("Load : " + args.test_data)
        motion_data = dataset['output_video'][:, :, :, :3]
        appearance_data = dataset['output_video'][:, :, :, -3:]
        label = dataset['output_label'][:]
    elif args.preprocessing == 1:
        dataset = DatasetDeepPhysUBFC()
        dataset = dataset()
        print("save : " + dir_path + "/preprocessing/train/UBFC_train_Data.hdf5")
    else:
        print('\nError! No such Data. Choose from preprocessing : 0.False 1.True')

    # Data Load
    mode = None
    if args.mode == 0:
        # Change Tensor & Split Data
        dataset = bp.dataset(A=appearance_data, M=motion_data, T=label)
        train_set, val_set = torch.utils.data.random_split(dataset,
                                                           [int(np.floor(len(dataset) * 0.7)),
                                                            int(np.ceil(len(dataset) * 0.3))])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        # Train
        train(model=model, train_loader=train_loader, val_loader=val_loader, criterion=loss_fn, optimizer=optimizer,
              model_path=args.checkpoint, epochs=args.epoch, device=device)
    elif args.mode == 1:
        # Change Tensor & Split Data
        dataset = bp.dataset(A=appearance_data, M=motion_data, T=label)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        # test
        test(model=model, test_loader=test_loader, check_model=args.check_model, device=device)
    else:
        print('\nError! No such Data. Choose from preprocessing : 0.Train 1.Test')

    mask_path = "/home/js/Desktop/Data/Pytorch_rppgs_save/preprocessing/mask/UBFC_mask_Data.hdf5"
    get_noise_estimates(mask_path)

