import copy
import datetime
import json
import math
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset.dataset_loader import dataset_loader
from log import log_info_time
from loss import loss_fn
from models import is_model_support, get_model, summary
from optim import optimizer
from torch.optim import lr_scheduler
from utils.dataset_preprocess import preprocessing
from utils.funcs import normalize, plot_graph, detrend
from utils.funcs import detrend
from heartpy import process
from sklearn.model_selection import KFold
import config as config
import wandb
from utils.image_preprocess import get_haarcascade

bpm_flag = False
K_Fold_flag = False
#Define Kfold Cross Validator
if K_Fold_flag:
    kfold = KFold(n_splits=5, shuffle=True)

wandb.init(project="SeqNet",entity="daeyeolkim")

now = datetime.datetime.now()
os.environ["CUDA_VISIBLE_DEVICES"]="9"

with open('params.json') as f:
    jsonObject = json.load(f)
    __PREPROCESSING__ = jsonObject.get("__PREPROCESSING__")
    __TIME__ = jsonObject.get("__TIME__")
    __MODEL_SUMMARY__ = jsonObject.get("__MODEL_SUMMARY__")
    options = jsonObject.get("options")
    params = jsonObject.get("params")
    hyper_params = jsonObject.get("hyper_params")
    model_params = jsonObject.get("model_params")
#
"""
Check Model Support
"""
is_model_support(model_params["name"], model_params["name_comment"])

'''
Generate preprocessed data hpy file 
'''
print("preprocessing")
if __PREPROCESSING__:
    if __TIME__:
        start_time = time.time()

    preprocessing(save_root_path=params["save_root_path"],
                  model_name=model_params["name"],
                  data_root_path=params["data_root_path"],
                  dataset_name=params["dataset_name"],
                  train_ratio=params["train_ratio"])
    if __TIME__:
        log_info_time("preprocessing time \t:", datetime.timedelta(seconds=time.time() - start_time))

'''
Load dataset before using Torch DataLoader
'''
print("Data Loader")
if __TIME__:
    start_time = time.time()

dataset = dataset_loader(save_root_path=params["save_root_path"],
                         model_name=model_params["name"],
                         dataset_name=params["dataset_name"],
                         option="train")
# test_dataset = dataset_loader(save_root_path=params["save_root_path"],
#                               model_name=model_params["name"],
#                               dataset_name=params["dataset_name"],
#                               option="test")
if not K_Fold_flag:
    dataset_len = len(dataset)
    train_len = int(np.floor(dataset_len * 0.84))
    val_len = int(np.floor(dataset_len * 0.1))
    test_len = dataset_len - train_len - val_len

    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_len,val_len,test_len])
if __TIME__:
    log_info_time("load train hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

if __TIME__:
    start_time = time.time()

if __TIME__:
    log_info_time("load test hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
    Call dataloader for iterate dataset
'''
if __TIME__:
    start_time = time.time()
if not K_Fold_flag:
    train_loader = DataLoader(train_dataset, batch_size=params["train_batch_size"],
                              shuffle=params["train_shuffle"])
    validation_loader = DataLoader(validation_dataset, batch_size=params["train_batch_size"],
                                   shuffle=params["train_shuffle"])
test_loader = DataLoader(test_dataset, batch_size=params["test_batch_size"],
                         shuffle=False)#params["test_shuffle"])
if __TIME__:
    log_info_time("generate dataloader time \t: ", datetime.timedelta(seconds=time.time() - start_time))
print("Set Model")
'''
Setting Learning Model
'''
if __TIME__:
    start_time = time.time()

# model = [get_model(model_params["name"])]
model = get_model(model_params["name"])
if torch.cuda.is_available():
    # os.environ["CUDA_VISIBLE_DEVICES"] = '9'
    # TODO: implement parallel training
    # if options["parallel_criterion"] :
    #     print(options["parallel_criterion_comment"])
    #     model = DataParallelModel(model, device_ids=[0, 1, 2])
    # else:
    #     model = DataParallel(model, output_device=0)
    # torch.cuda.set_device(int(options["set_gpu_device"]))

    # for mod in model[0]:
    #     mod.cuda()
    model.cuda()
else:
    # for mod in model[0]:
    #     mod = mod.to('cpu')
    model = model.to('cpu')
print("summary")
if __MODEL_SUMMARY__:

    summary(model,model_params["name"])

if __TIME__:
    log_info_time("model initialize time \t: ", datetime.timedelta(seconds=time.time() - start_time))

print("set loss")
'''
Setting Loss Function
'''
if __TIME__:
    start_time = time.time()
criterion = loss_fn(hyper_params["loss_fn"])
criterion2 = loss_fn("L1")

# if torch.cuda.is_available():
# TODO: implement parallel training
# if options["parallel_criterion"] :
#     print(options["parallel_criterion_comment"])
#     criterion = DataParallelCriterion(criterion,device_ids=[0, 1, 2])

if __TIME__:
    log_info_time("setting loss func time \t: ", datetime.timedelta(seconds=time.time() - start_time))
'''
Setting Optimizer
'''
print("set optim")
if __TIME__:
    start_time = time.time()
# optimizer = [optimizer(mod.parameters(),hyper_params["learning_rate"], hyper_params["optimizer"]) for mod in model[0]]
# scheduler = [lr_scheduler.ExponentialLR(optim,gamma=0.99) for optim in optimizer]
optimizer = optimizer(model.parameters(), hyper_params["learning_rate"], hyper_params["optimizer"])
scheduler = lr_scheduler.ExponentialLR(optimizer,gamma=0.99)

wandb.config = {
  "learning_rate": hyper_params["learning_rate"],
  "epochs": hyper_params["epochs"],
  "batch_size": params["train_batch_size"]
}


if __TIME__:
    log_info_time("setting optimizer time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
Model Training Step
'''
min_val_loss = 100.0
min_val_loss_model = None
if K_Fold_flag:
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = DataLoader(dataset,batch_size=params['train_batch_size'],sampler=train_subsampler,shuffle=True)
        validation_loader = DataLoader(dataset, batch_size=params['train_batch_size'], sampler=test_subsampler,shuffle=True)

        for epoch in range(hyper_params["epochs"]):
            with tqdm(train_loader, desc="Train ", total=len(train_loader)) as tepoch:
                # for mod in model[0]:
                #     mod.train()
                model.train()
                running_loss = 0.0
                bpm_loss = 0.0
                pcc_loss = 0.0
                i = 0
                r = 0
                cnt = 0
                # for inputs, target, _bpm in tepoch:
                for inputs, target in tepoch:
                    optimizer.zero_grad()
                    # [optim.zero_grad() for optim in optimizer]
                    tepoch.set_description(f"Train Epoch {epoch}")
                    if bpm_flag:
                        p, bpm ,att= model(inputs)
                    else:
                        # if model[0].__len__() == 1:
                        p = model(inputs)
                        # else:
                        #     noise_free_map = model[0][1](target)
                        #     p = model[0][0](inputs)

                    # _bpm = torch.reshape(torch.mean(_bpm, axis=1), (-1, 1))
                    if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "DeepPhys", "GCN","AxisNet"]:
                        loss0 = criterion(p, target)


                        # loss1 = criterion2(b['bpm'],d['bpm'])
                        # loss1 = criterion2(y,target[:,3:-3])#criterion(torch.gradient(torch.gradient(outputs,dim=1)[0],dim=1)[0],torch.gradient(torch.gradient(target,dim=1)[0],dim=1)[0])
                        if bpm_flag:
                            loss1 = criterion2(bpm, _bpm)

                            bpm_loss += loss1.item()
                            loss = loss0  + loss1
                        else:
                            loss = loss0
                    else:
                        loss_0 = criterion(y[:][0], target[:][0])
                        loss_1 = criterion(y[:][1], target[:][1])
                        loss_2 = criterion(y[:][2], target[:][2])
                        loss = loss_0 + loss_2 + loss_1

                    if ~torch.isfinite(loss):
                        continue

                    loss.backward()

                    pcc_loss += loss0.item()
                    running_loss += loss.item()
                    optimizer.step()

                    # [optim.step() for optim in optimizer]
                    tepoch.set_postfix(loss=running_loss / tepoch.__len__())
                wandb.log({"train_loss": running_loss / tepoch.__len__()})
                plt.clf()
                # plt.imshow(torch.permute(att[0],(1,2,3,0))[0].cpu().detach().numpy(),aspect='auto',origin='lower',interpolation='none')
                # plt.savefig('attention_map.png',figsize=(8,8))
                # wandb.log({
                #     "Attention Map": [
                #         wandb.Image('attention_map.png')
                #     ]
                # })
                if bpm_flag:
                    wandb.log({"train_pcc_loss": pcc_loss / tepoch.__len__()})
                    wandb.log({"train_bpm_loss": bpm_loss / tepoch.__len__()})


            if __TIME__ and epoch == 0:
                log_info_time("1 epoch training time \t: ", datetime.timedelta(seconds=time.time() - start_time))

            with tqdm(validation_loader, desc="Validation ", total=len(validation_loader)) as tepoch:
                model.eval()
                # for mod in model[0]:
                #     mod.eval()
                running_loss = 0.0
                bpm_loss = 0.0
                pcc_loss = 0.0
                with torch.no_grad():
                    # for inputs, target, _bpm in tepoch:
                    for inputs, target in tepoch:
                        tepoch.set_description(f"Validation")
                        if bpm_flag:
                            p, bpm, att = model(inputs)
                        else:
                            # if model[0].__len__() == 1:
                            p = model(inputs)
                            # else:
                            #     noise_free_map = model[0][1](target)
                            #     p = model[0][0](inputs)
                        # _bpm = torch.reshape(torch.mean(_bpm, axis=1), (-1, 1))
                        if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "DeepPhys", "GCN","AxisNet"]:
                            loss0 = criterion(p, target)

                            # loss1 = criterion2(y,target[:,3:-3])#criterion(torch.gradient(torch.gradient(outputs,dim=1)[0],dim=1)[0],torch.gradient(torch.gradient(target,dim=1)[0],dim=1)[0])
                            if bpm_flag:
                                loss1 = criterion2(bpm, _bpm)
                                bpm_loss += loss1.item()
                                loss = loss0 + loss1
                            else:
                                loss = loss0
                        else:
                            loss_0 = criterion(y[:][0], target[:][0])
                            loss_1 = criterion(y[:][1], target[:][1])
                            loss_2 = criterion(y[:][2], target[:][2])
                            loss = loss_0 + loss_2 + loss_1

                        if ~torch.isfinite(loss):
                            continue

                        pcc_loss += loss0.item()
                        running_loss += loss.item()
                        tepoch.set_postfix(loss=running_loss / tepoch.__len__())
                    # if min_val_loss > running_loss:  # save the train model
                    #     min_val_loss = running_loss
                    #     checkpoint = {'Epoch': epoch,
                    #                   'state_dict': model.state_dict(),
                    #                   'optimizer': optimizer.state_dict()}
                    #     torch.save(checkpoint, params["checkpoint_path"] + model_params["name"] + "/"
                    #                + params["dataset_name"] + "_" + str(epoch) + "_"
                    #                + str(min_val_loss) + '.pth')
                    #     min_val_loss_model = copy.deepcopy(model)
                wandb.log({"val_loss": running_loss / tepoch.__len__()})
                if bpm_flag:
                    wandb.log({"val pcc_loss": pcc_loss / tepoch.__len__()})
                    wandb.log({"val_bpm_loss": bpm_loss / tepoch.__len__()})

            if epoch + 1 == hyper_params["epochs"] or epoch % 5 == 0:
                if __TIME__ and epoch == 0:
                    start_time = time.time()
                # if epoch + 1 == hyper_params["epochs"]:
                    # model = min_val_loss_model
                with tqdm(test_loader, desc="test ", total=len(test_loader)) as tepoch:
                    # for mod in model[0]:
                    #     mod.eval()
                    model.eval()
                    inference_array = []
                    target_array = []
                    cnt = 0
                    running_loss = 0.0

                    pcc_loss = 0.0
                    with torch.no_grad():
                        # for inputs, target, _bpm in tepoch:
                        for inputs, target in tepoch:
                            tepoch.set_description(f"test")
                            # _bpm = torch.reshape(torch.mean(_bpm, axis=1), (-1, 1))
                            if bpm_flag:
                                p, bpm, att = model(inputs)
                            else:
                                # if model[0].__len__() == 1:
                                p = model(inputs)
                                # else:
                                #     noise_free_map = model[0][1](target)
                                #     p = model[0][0](inputs)
                            if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "DeepPhys", "GCN","AxisNet"]:
                                loss0 = criterion(p, target)

                                if bpm_flag:
                                    bpm_loss = 0.0
                                    loss1 = criterion2(bpm, _bpm)
                                    # print(bpm,_bpm)
                                    loss = loss0  + loss1
                                    bpm_loss += loss1.item()
                                else:
                                    loss = loss0
                            else:
                                loss_0 = criterion(y[:][0], target[:][0])
                                loss_1 = criterion(y[:][1], target[:][1])
                                loss_2 = criterion(y[:][2], target[:][2])
                                loss = loss_0 + loss_2 + loss_1

                            if ~torch.isfinite(loss):
                                continue

                            pcc_loss += loss0.item()
                            running_loss += loss.item()
                            tepoch.set_postfix(loss=running_loss / tepoch.__len__())
                            if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "GCN","AxisNet"]:
                                inference_array.extend(normalize(np.squeeze(p.cpu().numpy()[0])))
                                target_array.extend(normalize(target.cpu().numpy()[0]))
                            else:
                                inference_array.extend(p[:][0].cpu().numpy())
                                target_array.extend(target[:][0].cpu().numpy())
                            if tepoch.n == 0 and __TIME__:
                                save_time = time.time()
                        wandb.log({"test_loss": running_loss / tepoch.__len__()})
                        if bpm_flag:
                            wandb.log({"test_pcc_loss": pcc_loss / tepoch.__len__()})
                            wandb.log({"test_bpm_loss": bpm_loss / tepoch.__len__()})

                    # postprocessing
                    if model_params["name"] in ["DeepPhys"]:
                        inference_array = detrend(np.cumsum(inference_array), 100)
                        target_array = detrend(np.cumsum(target_array), 100)

                    # if __TIME__ and epoch == 0:
                    #     log_info_time("inference time \t: ", datetime.timedelta(seconds=save_time - start_time))
                    plt.clf()
                    plt = plot_graph(0, 300, target_array, inference_array, "original")
                    plt.savefig('graph.png', figsize=(16, 4))

                    wandb.log({
                        "Graph": [
                            wandb.Image('graph.png')
                        ]
                    })
                    # plt = plot_graph(0,300,target_array,detrend(inference_array,100),"filtered")
                    # wandb.log({"filtered":plt})
                    # plt = plot_graph(0,300,np.gradient(np.gradient(target_array[3:-3])),np.gradient(np.gradient(inference_array)),"gradient")
                    # wandb.log({"gradient": plt})
                    # print(np.gradient(target_array)-np.gradient(inference_array).mean(axis=0))

else:
    val_loss = 1.0
    test_flag = False
    for epoch in range(hyper_params["epochs"]):
        with tqdm(train_loader, desc="Train ", total=len(train_loader)) as tepoch:
            # for mod in model[0]:
            #     mod.train()
            model.train()
            running_loss = 0.0
            bpm_loss = 0.0
            pcc_loss = 0.0
            i = 0
            r = 0
            cnt = 0
            # for inputs, target, _bpm in tepoch:
            for inputs, target in tepoch:
                optimizer.zero_grad()
                # [optim.zero_grad() for optim in optimizer]
                tepoch.set_description(f"Train Epoch {epoch}")
                if bpm_flag:
                    p, bpm, att = model(inputs)
                else:
                    # if model[0].__len__() == 1:
                    p = model(inputs)
                    # else:
                    #     noise_free_map = model[0][1](target)
                    #     p = model[0][0](inputs)

                # _bpm = torch.reshape(torch.mean(_bpm, axis=1), (-1, 1))
                if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "DeepPhys", "GCN", "AxisNet"]:
                    loss0 = criterion(p, target)

                    # loss1 = criterion2(b['bpm'],d['bpm'])
                    # loss1 = criterion2(y,target[:,3:-3])#criterion(torch.gradient(torch.gradient(outputs,dim=1)[0],dim=1)[0],torch.gradient(torch.gradient(target,dim=1)[0],dim=1)[0])
                    if bpm_flag:
                        loss1 = criterion2(bpm, _bpm)

                        bpm_loss += loss1.item()
                        loss = loss0 + loss1
                    else:
                        loss = loss0
                else:
                    loss_0 = criterion(y[:][0], target[:][0])
                    loss_1 = criterion(y[:][1], target[:][1])
                    loss_2 = criterion(y[:][2], target[:][2])
                    loss = loss_0 + loss_2 + loss_1

                if ~torch.isfinite(loss):
                    continue

                loss.backward()

                pcc_loss += loss0.item()
                running_loss += loss.item()
                optimizer.step()

                # [optim.step() for optim in optimizer]
                tepoch.set_postfix(loss=running_loss / tepoch.__len__())
            wandb.log({"train_loss": running_loss / tepoch.__len__()})
            plt.clf()
            # plt.imshow(torch.permute(att[0],(1,2,3,0))[0].cpu().detach().numpy(),aspect='auto',origin='lower',interpolation='none')
            # plt.savefig('attention_map.png',figsize=(8,8))
            # wandb.log({
            #     "Attention Map": [
            #         wandb.Image('attention_map.png')
            #     ]
            # })
            if bpm_flag:
                wandb.log({"train_pcc_loss": pcc_loss / tepoch.__len__()})
                wandb.log({"train_bpm_loss": bpm_loss / tepoch.__len__()})

        if __TIME__ and epoch == 0:
            log_info_time("1 epoch training time \t: ", datetime.timedelta(seconds=time.time() - start_time))

        with tqdm(validation_loader, desc="Validation ", total=len(validation_loader)) as tepoch:
            model.eval()
            # for mod in model[0]:
            #     mod.eval()
            running_loss = 0.0
            bpm_loss = 0.0
            pcc_loss = 0.0
            with torch.no_grad():
                # for inputs, target, _bpm in tepoch:
                for inputs, target in tepoch:
                    tepoch.set_description(f"Validation")
                    if bpm_flag:
                        p, bpm, att = model(inputs)
                    else:
                        # if model[0].__len__() == 1:
                        p = model(inputs)
                        # else:
                        #     noise_free_map = model[0][1](target)
                        #     p = model[0][0](inputs)
                    # _bpm = torch.reshape(torch.mean(_bpm, axis=1), (-1, 1))
                    if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "DeepPhys", "GCN", "AxisNet"]:
                        loss0 = criterion(p, target)

                        # loss1 = criterion2(y,target[:,3:-3])#criterion(torch.gradient(torch.gradient(outputs,dim=1)[0],dim=1)[0],torch.gradient(torch.gradient(target,dim=1)[0],dim=1)[0])
                        if bpm_flag:
                            loss1 = criterion2(bpm, _bpm)
                            bpm_loss += loss1.item()
                            loss = loss0 + loss1
                        else:
                            loss = loss0
                    else:
                        loss_0 = criterion(y[:][0], target[:][0])
                        loss_1 = criterion(y[:][1], target[:][1])
                        loss_2 = criterion(y[:][2], target[:][2])
                        loss = loss_0 + loss_2 + loss_1

                    if ~torch.isfinite(loss):
                        continue

                    pcc_loss += loss0.item()
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=running_loss / tepoch.__len__())
                # if min_val_loss > running_loss:  # save the train model
                #     min_val_loss = running_loss
                #     checkpoint = {'Epoch': epoch,
                #                   'state_dict': model.state_dict(),
                #                   'optimizer': optimizer.state_dict()}
                #     torch.save(checkpoint, params["checkpoint_path"] + model_params["name"] + "/"
                #                + params["dataset_name"] + "_" + str(epoch) + "_"
                #                + str(min_val_loss) + '.pth')
                #     min_val_loss_model = copy.deepcopy(model)
            wandb.log({"val_loss": running_loss / tepoch.__len__()})
            cur_loss = running_loss / tepoch.__len__()
            if val_loss > cur_loss:
                val_loss = cur_loss
                test_flag = True
            if bpm_flag:
                wandb.log({"val pcc_loss": pcc_loss / tepoch.__len__()})
                wandb.log({"val_bpm_loss": bpm_loss / tepoch.__len__()})

        #if epoch + 1 == hyper_params["epochs"] or epoch % 5 == 0:
        if test_flag == True:
            test_flag = False
            if __TIME__ and epoch == 0:
                start_time = time.time()
            # if epoch + 1 == hyper_params["epochs"]:
            # model = min_val_loss_model
            with tqdm(test_loader, desc="test ", total=len(test_loader)) as tepoch:
                # for mod in model[0]:
                #     mod.eval()
                model.eval()
                inference_array = []
                target_array = []
                cnt = 0
                running_loss = 0.0

                pcc_loss = 0.0
                with torch.no_grad():
                    # for inputs, target, _bpm in tepoch:
                    for inputs, target in tepoch:
                        tepoch.set_description(f"test")
                        # _bpm = torch.reshape(torch.mean(_bpm, axis=1), (-1, 1))
                        if bpm_flag:
                            p, bpm, att = model(inputs)
                        else:
                            # if model[0].__len__() == 1:
                            p = model(inputs)
                            # else:
                            #     noise_free_map = model[0][1](target)
                            #     p = model[0][0](inputs)
                        if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "DeepPhys", "GCN", "AxisNet"]:
                            loss0 = criterion(p, target)

                            if bpm_flag:
                                bpm_loss = 0.0
                                loss1 = criterion2(bpm, _bpm)
                                # print(bpm,_bpm)
                                loss = loss0 + loss1
                                bpm_loss += loss1.item()
                            else:
                                loss = loss0
                        else:
                            loss_0 = criterion(y[:][0], target[:][0])
                            loss_1 = criterion(y[:][1], target[:][1])
                            loss_2 = criterion(y[:][2], target[:][2])
                            loss = loss_0 + loss_2 + loss_1

                        if ~torch.isfinite(loss):
                            continue

                        pcc_loss += loss0.item()
                        running_loss += loss.item()
                        tepoch.set_postfix(loss=running_loss / tepoch.__len__())
                        if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "GCN", "AxisNet"]:
                            inference_array.extend(normalize(np.squeeze(p.cpu().numpy()[0])))
                            target_array.extend(normalize(target.cpu().numpy()[0]))
                        else:
                            inference_array.extend(p[:][0].cpu().numpy())
                            target_array.extend(target[:][0].cpu().numpy())
                        if tepoch.n == 0 and __TIME__:
                            save_time = time.time()
                    wandb.log({"test_loss": running_loss / tepoch.__len__()})
                    if bpm_flag:
                        wandb.log({"test_pcc_loss": pcc_loss / tepoch.__len__()})
                        wandb.log({"test_bpm_loss": bpm_loss / tepoch.__len__()})

                # postprocessing
                if model_params["name"] in ["DeepPhys"]:
                    inference_array = detrend(np.cumsum(inference_array), 100)
                    target_array = detrend(np.cumsum(target_array), 100)

                # if __TIME__ and epoch == 0:
                #     log_info_time("inference time \t: ", datetime.timedelta(seconds=save_time - start_time))
                plt.clf()
                plt = plot_graph(0, 300, target_array, inference_array, "original")
                plt.savefig('graph.png', figsize=(16, 4))

                wandb.log({
                    "Graph": [
                        wandb.Image('graph.png')
                    ]
                })
                # plt = plot_graph(0,300,target_array,detrend(inference_array,100),"filtered")
                # wandb.log({"filtered":plt})
                # plt = plot_graph(0,300,np.gradient(np.gradient(target_array[3:-3])),np.gradient(np.gradient(inference_array)),"gradient")
                # wandb.log({"gradient": plt})
                # print(np.gradient(target_array)-np.gradient(inference_array).mean(axis=0))

                # plt = plot_graph(0,300,target_array,detrend(inference_array,100),"filtered")
                # wandb.log({"filtered":plt})
                # plt = plot_graph(0,300,np.gradient(np.gradient(target_array[3:-3])),np.gradient(np.gradient(inference_array)),"gradient")
                # wandb.log({"gradient": plt})
                # print(np.gradient(target_array)-np.gradient(inference_array).mean(axis=0))

