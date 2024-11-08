import os
import numpy as np
import torch
import torchvision
import argparse
import torch.nn.functional as F
import torch.nn as nn
from modules import ImageEncoder
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.multiprocessing
from dataset import ImageAugDataset
from tqdm import tqdm
import pickle
import pandas as pd
from collections import OrderedDict

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../data/M1/RS_s2_M1')
parser.add_argument('--bands', type=int, default=0)
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--resnet', type=str, default='resnet50')
parser.add_argument('--model_path', type=str, default='./ckpt')
parser.add_argument('--log', type=str, default='M1bands3-l8-log.txt')
parser.add_argument('--pkl', type=str, default="../data/Vis/train_on_M1bands3/M1bands3_M1_l8.pkl") # ../data/Vis/train_on_M1bands3/
parser.add_argument('--ckpt', type=str, default="M1bands3-l8_img_120.pth")


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda")

    ckpts = [args.ckpt]
    for ckpt in ckpts:
        model_fp = os.path.join(args.model_path, ckpt)
        # image encoder
        resnet = torchvision.models.resnet50(pretrained=False, num_classes=args.projection_dim)
        resnet.conv1 = nn.Conv2d(args.bands, 64, kernel_size=7, stride=2, padding=3,bias=False)
        dim_mlp = resnet.fc.weight.shape[1]
        img_encoder = ImageEncoder(resnet, args.projection_dim, dim_mlp).to(device)
 
        state_dict = torch.load(model_fp, map_location=device)
 
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module'
            new_state_dict[name] = v
        img_encoder.load_state_dict(new_state_dict)

        img_encoder = img_encoder.to(device)

        for name, param in img_encoder.named_parameters():
            #print(name)
            if 'fc' not in name:
                param.requires_grad = False
        
        img_encoder = nn.DataParallel(img_encoder)
        img_encoder.eval()
        dict = {}
                    
        for img_file in tqdm(os.listdir(args.data_path)):
            if args.bands == 6:
                img_input = np.load(os.path.join(args.data_path,img_file))
                img_input = np.transpose(img_input, (1,2,0))

            else:
                img_input = Image.open(os.path.join(args.data_path,img_file))

  
            ID = img_file.split("_")[-1].replace('.tif', '')
            
            toTensor = transforms.ToTensor()
            normalize = transforms.Normalize(
                                        mean=[0.1533, 0.1598, 0.1148], # s2 b3
                                        std=[0.1276, 0.1082, 0.0974]  # s2 b3
                                        )
            img_input = toTensor(img_input)
            img_input = normalize(img_input)
            img_input = img_input.unsqueeze(0).to(device)
            # input = torch.ones((1, 3, 32, 32)).to(device)
            output = img_encoder.module.encoder(img_input) # torch.Size([1, 128])
            feat = list(output.cpu().detach().numpy().flatten().astype(float))
            if os.path.exists(os.path.dirname(args.pkl)) is False:
                os.makedirs(os.path.dirname(args.pkl))
            dict[ID] = feat
            
        node_feats = pd.DataFrame(dict).T
        node_feats.index.name = 'geocode'
        node_feats.to_csv('../data/Vis/train_on_M1bands3/M1bands3_M1_s2_h2048.csv')

    
    
