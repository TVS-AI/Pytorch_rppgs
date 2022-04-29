import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.image_preprocess import video_normalize


class GCNDataset(Dataset):
    def __init__(self, video_data, label_data,bpm_data):
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                                           std=[0.229, 0.224, 0.225])])
        # self.video_data = np.resize(video_data,(video_data.shape[0],64,256,3))
        # self.video_data = video_data.repeat(2,axis=2).repeat(2,axis=3)
        self.video_data = video_data
        #smaller_img.repeat(2, axis=0).repeat(2, axis=1)
        # self.video_data = video_data[:,:,:,:,:]*2-1
        self.label = label_data
        self.bpm = bpm_data

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # video_data = torch.tensor(self.video_data[index],dtype=torch.float32)
        video_data = torch.tensor(np.transpose(self.video_data[index], (3,0,1,2)), dtype=torch.float32)
        label_data = torch.tensor(self.label[index], dtype=torch.float32)
        # bpm_data = torch.tensor(self.bpm[index],dtype=torch.float32)

        if torch.cuda.is_available():
            video_data = video_data.to('cuda')
            label_data = label_data.to('cuda')
            # bpm_data = bpm_data.to('cuda')

        return video_data, label_data#, bpm_data

    def __len__(self):
        return len(self.label)
