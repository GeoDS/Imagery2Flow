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
import logging

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../data/M1/l8_2020')
parser.add_argument('--bands', type=int, default=0)
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--model_path', type=str, default='./ckpt')
parser.add_argument('--log', type=str, default='log/M1bands3-l8.log')
parser.add_argument('--output_path', type=str) # ../data/Vis/train_on_M1bands3/M1bands3_M1_s2.csv'
parser.add_argument('--ckpt', type=str, default="M1bands3-l8_img_120.pth")


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda")

    # Setup logging
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(args.log, mode='a'), logging.StreamHandler()])
    logger = logging.getLogger('ImageEmbedding')
    logger.setLevel(logging.DEBUG)

    logger.info("-----------------------------------------")
    logger.info(f"Starting image embedding extraction with parameters:")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Bands: {args.bands}")
    logger.info(f"Projection dim: {args.projection_dim}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info("-----------------------------------------")

    ckpts = [args.ckpt]
    for ckpt in ckpts:
        model_fp = os.path.join(args.model_path, ckpt)
        logger.info(f"Loading model from {model_fp}")
        
        # image encoder
        encoder = torchvision.models.vit_l_16(pretrained=False, num_classes=args.projection_dim)
        dim_mlp = encoder.heads[-1].weight.shape[1]
        img_encoder = ImageEncoder(encoder, args.projection_dim, dim_mlp).to(device)
 
        state_dict = torch.load(model_fp, map_location=device)
 
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module'
            new_state_dict[name] = v
        img_encoder.load_state_dict(new_state_dict)

        img_encoder = img_encoder.to(device)

        for name, param in img_encoder.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        
        # img_encoder = nn.DataParallel(img_encoder)
        img_encoder.eval()
        dict = {}
                    
        logger.info("Starting image embedding extraction...")
        for img_file in tqdm(os.listdir(args.data_path)):
            if args.bands == 6:
                img_input = np.load(os.path.join(args.data_path,img_file))
                img_input = np.transpose(img_input, (1,2,0))
            else:
                img_input = Image.open(os.path.join(args.data_path,img_file))

            ID = img_file.split("_")[-1].replace('.tif', '')
            
            toTensor = transforms.ToTensor()
            normalize = transforms.Normalize(
                                            mean = [0.6247, 0.4714, 0.3403], # M1 l8
                                            std = [0.1945, 0.1398, 0.1141]
                                        )
            img_input = toTensor(img_input)
            img_input = normalize(img_input)
            img_input = img_input.unsqueeze(0).to(device)
            output = img_encoder.module.encoder(img_input)
            feat = list(output.cpu().detach().numpy().flatten().astype(float))
            dict[ID] = feat
            
        node_feats = pd.DataFrame(dict).T
        node_feats.index.name = 'geocode'
        node_feats.to_csv(args.output_path)
        logger.info(f"Embeddings saved to {args.output_path}")

    
    
