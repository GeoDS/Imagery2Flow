import pandas as pd
import numpy as np
import torch
import os
import dgl
import pickle

import warnings

def load_nids_dataset(node_feats_path='../data/Vis/default.csv', year=2020, fprefix='CommutingFlow_', region='default', mappath='../data/CensusTract2020/nodeid_geocode_mapping.csv'):
    
    region_prefix = region.split('t')[0]
    nid_dir = f"../data/Nid/{region_prefix}/"

    # Load the node IDs d
    train_nids = pd.read_csv(f'{nid_dir}train_nids_{region}.csv', dtype={'geocode': 'string'})
    val_nids = pd.read_csv(f'{nid_dir}valid_nids_{region}.csv', dtype={'geocode': 'string'})
    test_nids = pd.read_csv(f'{nid_dir}test_nids_{region}.csv', dtype={'geocode': 'string'})
    all_nids = pd.read_csv(f'{nid_dir}all_nids_{region_prefix}.csv', dtype={'geocode': 'string'})
    
    # Load and prepare the mapping table
    mapping_table = pd.read_csv(mappath.replace(".csv", f"_{region_prefix}.csv"), dtype={
    'geocode': 'string'})

    # Load OD flows and convert geocodes to node IDs
    flow_dir='../data/LODES/'
    odflows_file = f'{flow_dir}{fprefix}{region_prefix}_{year}gt10.csv'
    odflows = pd.read_csv(odflows_file, dtype={
        'w_geocode': 'string',
        'h_geocode': 'string'
    })
    odflows = geocode_to_nodeid(odflows, mapping_table)    
    # Load and process node features
    node_feats = pd.read_csv(node_feats_path, dtype={
    'geocode': 'string'})
    node_feats['geocode'] = mapping_table.set_index('geocode').loc[node_feats['geocode']].values # map census tract to node id
    node_feats = node_feats.rename(columns={'geocode': 'nid'}).set_index('nid').sort_index()
   
    node_feats = (node_feats - node_feats.mean()) / node_feats.std()

    # Load and process adjacency matrix
    adjpath = f'../data/CensusTract2020/adjacency_matrix_bycar_m_{region_prefix}.csv'
    ct_adj = pd.read_csv(adjpath, dtype={
    'Unnamed: 0': 'string'}).set_index('Unnamed: 0')
    
    ct_inorder = mapping_table.sort_values(by='node_id')['geocode']
    ct_adj = ct_adj.loc[ct_inorder, ct_inorder.astype(str)].fillna(0)
    ct_adj = ct_adj / ct_adj.max().max() # min is 0
    
    mapping_table = mapping_table.set_index('geocode')
    # Compile data dictionary
    data = {
        # 'train_on': train_on,
        'train_nids': mapping_table.loc[train_nids['geocode']].values.ravel(),
        'valid_nids': mapping_table.loc[val_nids['geocode']].values.ravel(),
        'test_nids': mapping_table.loc[test_nids['geocode']].values.ravel(),
        'all_nids': mapping_table.loc[all_nids['geocode']].values.ravel(),
        'odflows': odflows[['src', 'dst', 'count', 'dis_m']].values,
        'num_nodes': ct_adj.shape[0],
        'node_feats': node_feats.values,
        'weighted_adjacency': ct_adj.values
    }
    return data


def geocode_to_nodeid(dataframe, mapping_table):
    df = dataframe.copy()
    mapping = mapping_table.copy()
    mapping.set_index('geocode', inplace=True)
    df['src'] = mapping.loc[df['h_geocode']].values
    df['dst'] = mapping.loc[df['w_geocode']].values

    return df[['src', 'dst', 'count','dis_m']]
    
def nodeid_to_geocode(dataframe, region):
    df = dataframe.copy()
    region_prefix = region.split('t')[0]
    mapping = pd.read_csv(f'../data/CensusTract2020/nodeid_geocode_mapping_{region_prefix}.csv').copy()           
    mapping.set_index('node_id', inplace=True)
    df['h_geocode'] = mapping.loc[df['src']].values
    df['w_geocode'] = mapping.loc[df['dst']].values

    return df[['h_geocode', 'w_geocode', 'count', 'prediction']]

def build_graph_from_matrix(adj_matrix, node_feats, device='cpu'):
    dst, src = adj_matrix.nonzero()
    edge_weights = torch.tensor(adj_matrix[adj_matrix.nonzero()]).float().view(-1, 1)
    g = dgl.DGLGraph()
    g = g.to('cuda:0')
    g.add_nodes(adj_matrix.shape[0])
    g.add_edges(src, dst, {'d': edge_weights})
    g.ndata['attr'] = torch.from_numpy(node_feats).to(device)

    return g
    


def evaluateOne(model, g, trip_od, trip_volume, output_nodes):
    with torch.no_grad():
        node_embedding = model(g)
        log_prediction = model.predict_edge(node_embedding, trip_od, output_nodes)
        prediction = exp_transform(log_prediction)
        y = trip_volume.float().view(-1, 1)
        rmse = RMSE(prediction, y)
        mae = MAE(prediction, y)
        cpc = CPC(prediction, y)
       

    return rmse.item(), mae.item(), cpc.item()

def evaluateOutput(model, g, trip_od, trip_volume, output_nodes, region, prefix, train_on):
    with torch.no_grad():
        node_embedding = model(g)
        log_prediction = model.predict_edge(node_embedding, trip_od, output_nodes)
        prediction = exp_transform(log_prediction)
        y = trip_volume.float().view(-1, 1)

        rmse = RMSE(prediction, y)
        mae = MAE(prediction, y)
        cpc = CPC(prediction, y)
        
        result = pd.DataFrame(torch.cat((trip_od, y, prediction), 1).cpu().numpy(), 
                              columns=['src','dst','count','prediction'])
        result = nodeid_to_geocode(result,region)
        # train_on = 'M1bands6' or 'M1bands6_0.2'
        if os.path.exists(os.path.join('outputs', train_on.split('_')[0]))==False:
          os.makedirs(os.path.join('outputs', train_on.split('_')[0]))
        result.to_csv(os.path.join('outputs', train_on.split('_')[0], 
                                   train_on+'_'+region+'_prediction_'+prefix+'.csv')) #'outputs/M1bands3_'+region+'_prediction_'+prefix+'.csv')
        

    return rmse.item(), mae.item(), cpc.item()


def log_transform(y):
    return torch.log2(y)

def exp_transform(scaled_y):
    output = torch.clamp(scaled_y, min=1)
    return torch.pow(2, output)

def RMSE(y_hat, y):
    return torch.sqrt(torch.mean((y_hat - y)**2))

def MAE(y_hat, y):
    abserror = torch.abs(y_hat - y)
    return torch.mean(abserror)

def CPC(y_hat, y):
    return 2 * torch.sum(torch.min(y_hat, y)) / (torch.sum(y_hat) + torch.sum(y))

def CPC_(y, y_hat):
    return 2 * np.minimum(y_hat, y).sum() / (y_hat.sum() + y.sum()) 