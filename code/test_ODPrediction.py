import pandas as pd
import numpy as np
import torch
import logging 

from dataset import utils
from modules.gnn import MyModelBlock
import dgl
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='log/default.log') # model state of city A 
parser.add_argument('--node_feats_path', type=str, default='../data/Vis/train_on_M1bands3/M1bands3_M2_l8.csv') # visual features of city B
parser.add_argument('--region', type=str, default='M2') # city B
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

    # Setup logging
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(test_args.log, mode='a'), logging.StreamHandler()])
    logger = logging.getLogger('#layers{}_emb{}'.format(test_args.num_hidden_layers, test_args.embedding_size))
    logger.setLevel(logging.DEBUG)

    # load data
    region = test_args.region
    data = utils.load_nids_dataset(year=test_args.year,node_feats_path=test_args.node_feats_path, region=region)

    logger.info("-----------------------------------------")
    logger.info(f"Starting testing with parameters:")
    logger.info(f"Region: {region}")
    logger.info(f"Year: {test_args.year}")
    logger.info(f"Node features path: {test_args.node_feats_path}")
    logger.info(f"Number of hidden layers: {test_args.num_hidden_layers}")
    logger.info(f"Embedding size: {test_args.embedding_size}")
    logger.info("-----------------------------------------")

    logger.info("----------------------------------------- "+region+" "+test_args.year+" Test all_nids")
    all_nids = data['all_nids'] 

    odflows = data['odflows']
    all_nids_d = np.unique(odflows[np.isin(odflows[:, 0], all_nids)][:,1]).astype('int64')
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
    logger.info(f"Loading model from {model_state_file}")
    model.load_state_dict(torch.load(model_state_file)['state_dict']) 
    model.eval()
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            test_dataloader
    ):
        with torch.no_grad():
            node_embedding = model(blocks).detach().cpu().numpy()
            origin_nids = output_nodes.cpu()[np.isin(output_nodes.cpu(), all_nids)]

            trip_od_test = torch.from_numpy(odflows[:, :2].astype(np.int64)).to(device)
            # scaled_dism_test = torch.from_numpy(odflows[:, 3].astype(float)).to(device)

            trip_volume_test = torch.from_numpy(odflows[:, 2].astype(float)).to(device)
            log_trip_volume_test = utils.log_transform(trip_volume_test)

            # loss = model.get_loss(output_nodes, trip_od_test, log_trip_volume_test, blocks)

        rmse, mae, cpc = utils.evaluateOutput(model, blocks, trip_od_test, trip_volume_test, output_nodes, region, prefix, data['train_on'])
        
        logger.info("-----------------------------------------")
        logger.info(f'Test | Bilinear Decoder '
                 f'RMSE: {rmse:.4f} - MAE: {mae:.4f} - '
                 f'CPC: {cpc:.4f}')

    indices_o = [torch.where(output_nodes == b)[0] for b in trip_od_test[:,0]]
    flattened_o = torch.cat(indices_o).cpu().numpy()
    indices_d = [torch.where(output_nodes == b)[0] for b in trip_od_test[:,1]]
    flattened_d = torch.cat(indices_d).cpu().numpy()
    scaled_dism_test = odflows[:, 3]
    # construct edge feature

    X_test = np.concatenate([node_embedding[flattened_o], node_embedding[flattened_d],scaled_dism_test.reshape(-1,1)], axis=1)
    y_test = odflows[:, 2]

    with open('./models/lgbm_{}.txt'.format(args.log.strip('log/').strip('.log')), 'rb') as file:
        gbm = pickle.load(file)
    y_gbm = gbm.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_gbm))
    mae = mean_absolute_error(y_test, y_gbm)
    logger.info(f"Test - | LGBM Decoder RMSE: {rmse:.4f} - MAE: {mae:.4f} - CPC: {utils.CPC_(y_test, y_gbm)  :.4f}")


if __name__ == "__main__":
    args = parser.parse_args()
    
    test(args)
