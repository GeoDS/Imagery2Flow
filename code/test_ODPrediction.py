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
parser.add_argument('--node_feats_path', type=str, default='../data/Vis/train_on_M1bands3/M1bands3_M1_l8_h2048.csv') 
parser.add_argument('--region', type=str, default='default') # 'NYC/CHI/MSN'
parser.add_argument('--year', type=str, default='2020') 
parser.add_argument('--device', type=str, default = 'cuda:0')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default = 1e-5)
parser.add_argument('--grad_norm', type=float, default=1.0)
parser.add_argument('--evaluate_every', type=int, default=5)

parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--embedding_size', type=int, default=512)

def test(test_args):
    # device
    device = torch.device(test_args.device)

    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(test_args.log, mode='a'), logging.StreamHandler()])
    logger = logging.getLogger('#layers{}_emb{}'.format(test_args.num_hidden_layers, test_args.embedding_size))
    logger.setLevel(logging.DEBUG)
    # load data
    region = test_args.region
    data = utils.load_nids_dataset(year=test_args.year,node_feats_path=test_args.node_feats_path, region=region)

    logger.info("----------------------------------------- "+region+" "+test_args.year+" Test all_nids")
    all_nids = data['all_nids'] 

    odflows = data['odflows']
    all_nids_d = np.unique(odflows[np.isin(odflows[:, 0], all_nids)][:,1])
    all_nids_od =  np.unique(np.append(all_nids, all_nids_d, axis=0))
    
    node_feats = data['node_feats']

    ct_adj = data['weighted_adjacency']
    num_nodes = data['num_nodes']

    model = MyModelBlock(num_nodes, in_dim = node_feats.shape[1], h_dim = test_args.embedding_size, num_hidden_layers=test_args.num_hidden_layers, device=device)
    g = utils.build_graph_from_matrix(ct_adj, node_feats.astype(np.float32), device)  

    model.to(device)
       
    g.to(device)

    # minibatch
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(test_args.num_hidden_layers+1)

    test_dataloader = dgl.dataloading.DataLoader(
        g, torch.from_numpy(all_nids_od).to(device), sampler,
        batch_size=len(all_nids_od),
        shuffle=True,
        drop_last=False,
        num_workers=0) # 

    
    # training recorder
    model_state_file = './ckpt/{}_layers{}_emb{}.pth'.format(test_args.log.strip('log/').strip('.log'),test_args.num_hidden_layers, test_args.embedding_size)
    
    li = test_args.log.split("_")
    prefix = li[2] + "_" + li[1] + "_#layers{}_emb{}".format(test_args.num_hidden_layers, test_args.embedding_size) + "_" + li[3] + "_y" + test_args.year
      
    # Test:
    model.load_state_dict(torch.load(model_state_file)['state_dict']) 
    model.eval()
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            test_dataloader
    ):
        with torch.no_grad():
            origin_train_nids = output_nodes.cpu()[np.isin(output_nodes.cpu(), all_nids)]

            condition1 = np.isin(odflows[:, 0], origin_train_nids)
            condition2 = np.isin(odflows[:, 1], output_nodes.cpu())

            combined_condition = condition1 & condition2

            trip_od_test = torch.from_numpy(odflows[combined_condition][:, :2].astype(np.int64)).to(device)

            trip_volume_test = torch.from_numpy(odflows[combined_condition][:, -1].astype(float)).to(device)
            log_trip_volume_test = utils.log_transform(trip_volume_test)

            loss = model.get_loss(output_nodes, trip_od_test, log_trip_volume_test, blocks)

        rmse, mae, cpc = utils.evaluateOutput(model, blocks, trip_od_test, trip_volume_test, output_nodes, region, prefix, data['train_on'])
        
        logger.info("-----------------------------------------")
        logger.info(f'Test - Loss: {loss:.4f} | '
                 f'RMSE: {rmse:.4f} - MAE: {mae:.4f} - '
                 f'CPC: {cpc:.4f}')


if __name__ == "__main__":
    args = parser.parse_args()
    
    test(args)
