import pandas as pd
import numpy as np
import torch

import logging 

from dataset import utils
from modules.gnn import MyModelBlock
import dgl


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='log/default.log') # 'log/default.log'
parser.add_argument('--node_feats_path', type=str, default='../data/Vis/train_on_M1bands3/M1bands3_M1_l8.pkl') 
parser.add_argument('--region', type=str, default='M1') 
parser.add_argument('--year', type=str, default='2020') 
parser.add_argument('--device', type=str, default = 'cuda:0')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default = 1e-5)
parser.add_argument('--grad_norm', type=float, default=1.0)
parser.add_argument('--evaluate_every', type=int, default=5)

parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--embedding_size', type=int, default=512)

def adjust_learning_rate_warmup(optimizer, epoch, warmup_epochs, initial_lr, max_lr):
    if epoch < warmup_epochs:
        lr = initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            if 'name' in param_group and param_group['name'] == 'noise_sigma':
                continue
            param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            if 'name' in param_group and param_group['name'] == 'noise_sigma':
                continue
            param_group['lr'] = max_lr

def train(train_args):
    # device
    device = torch.device(train_args.device)
   
    # logger
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(train_args.log, mode='a'), logging.StreamHandler()])
    logger = logging.getLogger('#layers {}, emb {}'.format(train_args.num_hidden_layers, train_args.embedding_size))
    logger.setLevel(logging.DEBUG)

    torch.manual_seed(2024)
    np.random.seed(2024)

    region = train_args.region
    data = utils.load_nids_dataset(year=train_args.year,node_feats_path=train_args.node_feats_path, region=region)

    train_nids = data['train_nids']
    valid_nids = data['valid_nids']
    test_nids = data['test_nids']
    # all_nids = data['all_nids']

    odflows = data['odflows']

    train_nids_d = np.unique(odflows[np.isin(odflows[:, 0], train_nids)][:,1])
    train_nids_od =  np.unique(np.append(train_nids, train_nids_d, axis=0))
    valid_nids_d = np.unique(odflows[np.isin(odflows[:, 0], valid_nids)][:,1])
    valid_nids_od =  np.unique(np.append(valid_nids, valid_nids_d, axis=0))
    test_nids_d = np.unique(odflows[np.isin(odflows[:, 0], test_nids)][:,1])
    test_nids_od =  np.unique(np.append(test_nids, test_nids_d, axis=0))

    node_feats = data['node_feats']
    init_noise_sigma = np.std(np.log10(odflows[np.isin(odflows[:, 0], train_nids)][:, 2]))
    print("init_noise_sigma ", init_noise_sigma)

    ct_adj = data['weighted_adjacency']

    num_nodes = data['num_nodes']

    model = MyModelBlock(num_nodes, in_dim = node_feats.shape[1], h_dim = train_args.embedding_size, num_hidden_layers=train_args.num_hidden_layers,init_noise_sigma = init_noise_sigma, device=device)
    g = utils.build_graph_from_matrix(ct_adj, node_feats.astype(np.float32), device)  
    
    model.to(device)
    
    g.to(device)

    # minibatch
    batch_size = 512
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(train_args.num_hidden_layers+1)
    train_dataloader = dgl.dataloading.DataLoader(
        g, torch.from_numpy(train_nids_od).to(device), sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
    valid_dataloader = dgl.dataloading.DataLoader(
        g, torch.from_numpy(valid_nids_od).to(device), sampler,
        batch_size=len(valid_nids_od),
        shuffle=True,
        drop_last=False)
    test_dataloader = dgl.dataloading.DataLoader(
        g, torch.from_numpy(test_nids_od).to(device), sampler,
        batch_size=len(test_nids_od), 
        shuffle=True,
        drop_last=False)
  
    model_state_file = './ckpt/{}_layers{}_emb{}.pth'.format(train_args.log.strip('log/').strip('.log'),train_args.num_hidden_layers, train_args.embedding_size)
    best_rmse = float('inf')

    initial_lr = 1e-6
    warmup_epochs = 5
    early_stopping_patience = 20
    best_val_loss = float('inf')
    early_stopping_counter = 0
    noise_var = AverageMeter()
    criterion_params = list(model.criterion.parameters())
    # Get the IDs of criterion parameters
    criterion_param_ids = list(map(id, criterion_params))
    # Filter out criterion parameters from model parameters
    model_params = filter(lambda p: id(p) not in criterion_param_ids, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': model_params, 'lr': train_args.lr},
        {'params': criterion_params, 'lr': 1e-2}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1)
    
    
    for epoch in range(train_args.max_epochs):
        model.train()
        adjust_learning_rate_warmup(optimizer, epoch, warmup_epochs, initial_lr, train_args.lr)
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            # clear gradients
            optimizer.zero_grad()
            origin_train_nids = output_nodes.cpu()[np.isin(output_nodes.cpu(), train_nids)]

            condition1 = np.isin(odflows[:, 0], origin_train_nids)
            condition2 = np.isin(odflows[:, 1], output_nodes.cpu())

            combined_condition = condition1 & condition2

            trip_od = torch.from_numpy(odflows[combined_condition][:, :2].astype(np.int64)).to(device)

            log_trip_volume = utils.log_transform(torch.from_numpy(odflows[combined_condition][:, -1].astype(float))).to(device)
            loss = model.get_loss(output_nodes, trip_od, log_trip_volume, blocks)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_norm)
            optimizer.step()
            
        scheduler.step()
        if logger.level == logging.DEBUG:
            model.eval()
            for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
            ):
                with torch.no_grad():

                    origin_train_nids = output_nodes.cpu()[np.isin(output_nodes.cpu(), train_nids)]

                    condition1 = np.isin(odflows[:, 0], origin_train_nids)
                    condition2 = np.isin(odflows[:, 1], output_nodes.cpu())

                    combined_condition = condition1 & condition2

                    trip_od = torch.from_numpy(odflows[combined_condition][:, :2].astype(np.int64)).to(device)

                    trip_volume = torch.from_numpy(odflows[combined_condition][:, -1].astype(float)).to(device)
                    log_trip_volume = utils.log_transform(trip_volume)
                    loss = model.get_loss(output_nodes, trip_od, log_trip_volume, blocks)
                rmse, mae, cpc = utils.evaluateOne(model, blocks, trip_od, trip_volume, output_nodes)
            
                logger.debug(f'Epoch: {epoch:04d} - Train - Loss: {loss:.4f} | '
                 f'RMSE: {rmse:.4f} - MAE: {mae:.4f} - '
                 f'CPC: {cpc:.4f}')


        # Valid
        val_loss = 0
        if epoch % train_args.evaluate_every == 0 or epoch == (train_args.max_epochs-1):
            model.eval()
            for it, (input_nodes, output_nodes, blocks) in enumerate(
            valid_dataloader
            ):

                with torch.no_grad():

                    origin_valid_nids = output_nodes.cpu()[np.isin(output_nodes.cpu(), valid_nids)]

                    condition1 = np.isin(odflows[:, 0], origin_valid_nids)
                    condition2 = np.isin(odflows[:, 1], output_nodes.cpu())

                    combined_condition = condition1 & condition2

                    trip_od_valid = torch.from_numpy(odflows[combined_condition][:, :2].astype(np.int64)).to(device)
                    trip_volume_valid = torch.from_numpy(odflows[combined_condition][:, -1].astype(float)).to(device)
                    log_trip_volume_valid = utils.log_transform(trip_volume_valid)
                    loss = model.get_loss(output_nodes, trip_od_valid, log_trip_volume_valid, blocks)

                rmse, mae, cpc = utils.evaluateOne(model, blocks, trip_od_valid, trip_volume_valid, output_nodes)
        
                logger.info("-----------------------------------------")
                logger.info(f'Epoch: {epoch:04d} - Validation - Loss: {loss:.4f} | '
                 f'RMSE: {rmse:.4f} - MAE: {mae:.4f} - '
                 f'CPC: {cpc:.4f}')
                if rmse < best_rmse:
                    best_rmse = rmse
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'rmse': rmse, 'mae': mae, 'cpc': cpc}, model_state_file)
                    logger.info('Best RMSE found on epoch {}'.format(epoch))
                val_loss = loss
                       
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0  
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    logger.info("Early stopping at epoch {}.".format(epoch))
                    break
            logger.info("-----------------------------------------")


    
    # Test:
    li = train_args.log.split("_")
    prefix = li[2] + "_" + li[1] + "_#layers{}_emb{}".format(train_args.num_hidden_layers, train_args.embedding_size) + "_" + li[3]
    train_on = li[0].replace("log/","").replace("bands3","")
    logger.info("----------------------------------------- "+region+" 0.2 Test")
    model.eval()
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            test_dataloader
    ):
        with torch.no_grad():
            origin_test_nids = output_nodes.cpu()[np.isin(output_nodes.cpu(), test_nids)]
            condition1 = np.isin(odflows[:, 0], origin_test_nids)
            condition2 = np.isin(odflows[:, 1], output_nodes.cpu())
            combined_condition = condition1 & condition2
            trip_od_test = torch.from_numpy(odflows[combined_condition][:, :2].astype(np.int64)).to(device)
            trip_volume_test = torch.from_numpy(odflows[combined_condition][:, -1].astype(float)).to(device)
            log_trip_volume_test = utils.log_transform(trip_volume_test)

            loss = model.get_loss(output_nodes, trip_od_test, log_trip_volume_test, blocks)
        rmse, mae, cpc= utils.evaluateOutput(model, blocks, trip_od_test, trip_volume_test, output_nodes, region, prefix, train_on+"_0.2")
        
        # report
        logger.info("-----------------------------------------")
        logger.info(f'Epoch: {epoch:04d} - Test - Loss: {loss:.4f} | '
                 f'RMSE: {rmse:.4f} - MAE: {mae:.4f} - '
                 f'CPC: {cpc:.4f}')
        
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    args = parser.parse_args()
    
    train(args)  
