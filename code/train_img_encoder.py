import os
import torch
import torchvision
import argparse
import torch_geometric.data
import math
import torch.nn as nn
from modules import *
import logging

from dataset import ImageAugDataset

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../data/M1/l8_2020') 
parser.add_argument('--bands', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--resnet', type=str, default='resnet50')
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--total_epoch', type=int, default=120)
parser.add_argument('--model_path', type=str, default='./ckpt')
parser.add_argument('--schedule', default=[90, 110], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', type=bool, default=False)
parser.add_argument('--log', type=str, default='log/M1bands3-l8.log')


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.total_epoch))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            if epoch >= milestone:
                lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(args.log, mode='a'), logging.StreamHandler()])
    logger = logging.getLogger('ImageEncoder')
    logger.setLevel(logging.DEBUG)

    # dataset
    train_dataset = ImageAugDataset(path=args.data_path)
    
    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=6)

    # image encoder
    encoder = torchvision.models.vit_l_16(pretrained=False, num_classes=args.projection_dim)
    dim_mlp = encoder.heads[-1].weight.shape[1]
    img_encoder = ImageEncoder(encoder, args.projection_dim, dim_mlp).to(device)

    img_encoder = nn.DataParallel(img_encoder)
    img_encoder.train()

    # optimizer
    img_optimizer = torch.optim.Adam(img_encoder.parameters(), lr=args.lr)

    # loss
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    prefix = args.log.strip('log/').strip('.log')
    epoch_losses = []

    logger.info("-----------------------------------------")
    logger.info(f"Starting training with parameters:")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Bands: {args.bands}")
    logger.info(f"Projection dim: {args.projection_dim}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Total epochs: {args.total_epoch}")
    logger.info("-----------------------------------------")

    for epoch in range(1, args.total_epoch + 1):
        img_optimizer = adjust_learning_rate(img_optimizer, epoch, args)
        epoch_loss = 0
        
        for step, data in enumerate(train_loader):
            img_optimizer.zero_grad()
            
            img1, img2 = data[0], data[1]

            img1 = img1.to(device)
            img2 = img2.to(device)

            h_img1, h_img2, z_img1, z_img2 = img_encoder(img1, img2)

            loss = criterion(z_img1, z_img2)
            epoch_loss += loss.item()

            loss.backward()
            img_optimizer.step()

            if step % 10 == 0:
                logger.debug(f'Epoch: {epoch:03d} | Step: {step:03d} | Loss: {loss.item():.4f}')

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)
        logger.info(f'Epoch: {epoch:03d} | Average Loss: {avg_epoch_loss:.4f}')

    out_pth = os.path.join(args.model_path, "{}_img_{}.pth".format(prefix, args.total_epoch))
    torch.save(img_encoder.state_dict(), out_pth)
    logger.info(f"Model saved to {out_pth}")
    
    plt.figure()                   
    plt.plot(epoch_losses,'b',label = 'loss', linewidth=0.5)       
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(ticks=np.arange(0, len(train_loader)*args.total_epoch, step=len(train_loader)*20), labels=np.arange(0, args.total_epoch, step=20))
    plt.legend()        
    plt.savefig(os.path.join('./log',"{}_loss.jpg".format(prefix)))
    logger.info(f"Loss plot saved to {os.path.join('./log', prefix + '_loss.jpg')}")
