import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from torch.nn.modules.loss import _Loss

class MyModelBlock(nn.Module):

    def __init__(self, num_nodes, in_dim, h_dim, weighted=False, init_noise_sigma=0.2, num_hidden_layers=1,device='cpu'):
        super().__init__()

        self.gat = GAT(num_nodes, in_dim, h_dim, h_dim, num_hidden_layers, device) # GAT for origin node
        self.bilinear = nn.Bilinear(h_dim, h_dim, 1)
        self.activation = nn.ReLU()
        self.criterion = BMCLoss(init_noise_sigma, weighted=weighted)

    def forward(self, g):
        return self.gat.forward(g)
    
    def get_loss(self,output_nodes, trip_od, scaled_trip_volume, g):
        node_embedding = self.forward(g)
        edge_prediction = self.predict_edge(node_embedding, trip_od, output_nodes)
        edge_predict_loss = self.criterion(edge_prediction.to(torch.float32), scaled_trip_volume.to(torch.float32))

        return edge_predict_loss 
    
    def predict_edge(self, node_embedding, trip_od, output_nodes):

        indices_o = [torch.where(output_nodes == b)[0] for b in trip_od[:,0]]
        flattened_o = torch.cat(indices_o)
        indices_d = [torch.where(output_nodes == b)[0] for b in trip_od[:,1]]
        flattened_d = torch.cat(indices_d)
        # construct edge feature
        src_emb = node_embedding[flattened_o]
        dst_emb = node_embedding[flattened_d]
        # get predictions
        edge = self.bilinear(src_emb, dst_emb)
        edge = self.activation(edge)
        return edge

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma, weighted=False):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(pred, target, noise_var)
        return loss


def bmc_loss(pred, target, noise_var, weighted=False):
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var).detach()
    # gamma = 2
    if weighted:
        weights = (1 / target) # ** gamma
        loss = (loss * weights).mean()

    return loss

class GAT(nn.Module):

    def __init__(self, num_nodes, in_dim, h_dim, out_dim, num_hidden_layers=1, device='cpu'):
        # initialize super class
        super().__init__()

        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.device = device

        self.build_model()
    
    def build_model(self):
        self.layers = nn.ModuleList()
        
        i2h = GATLayer(self.in_dim, self.h_dim)
        self.layers.append(i2h)
       
        for i in range(self.num_hidden_layers):
            h2h = GATLayer(self.h_dim, self.h_dim)
            self.layers.append(h2h)

    def forward(self, blocks):
        h = blocks[0].srcdata['attr']
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(h, block)
        return h
    
class GATLayer(nn.Module):
    def __init__(self, in_ndim, out_ndim, in_edim=1, out_edim=1):
        super(GATLayer, self).__init__()
        self.fcV = nn.Linear(in_edim, out_edim, bias=False)
        self.fcW = nn.Linear(in_ndim, out_ndim, bias=False)
        self.fcU = nn.Linear(in_ndim, out_ndim, bias=False)

        self.attn_fc = nn.Linear(2 * out_ndim + out_edim, 1, bias=False)

        self.activation = F.relu


    def edge_feat_func(self, edges):
        return {'t': self.fcV(edges.data['d'].to('cuda:0'))}

    def edge_attention(self, edges):
        
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['t']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        
        return {'z': edges.src['z'],'e': edges.data['e']}

    def reduce_func(self, nodes):
     
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
       
        z_neighbor = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        z_i = nodes.data['z_i']
  
        h = self.activation(z_i + z_neighbor)
        return {'h': h}

    def forward(self, h, g):

        g.apply_edges(self.edge_feat_func)
        z = self.fcW(h) 
        g.srcdata['z'] = z 
        g.dstdata['z'] = z[:g.number_of_dst_nodes()]
        z_i = self.fcU(h) 
        g.srcdata['z_i'] = z_i 
        g.dstdata['z_i'] = z_i[:g.number_of_dst_nodes()]

        g.apply_edges(self.edge_attention)

        g.update_all(self.message_func, self.reduce_func)

        return g.dstdata.pop('h')


def MSE(y_hat, y, limit=20000):
    if y_hat.shape[0] < limit:
        return F.mse_loss(y_hat, y)
    else:
        mse = 0
        for i in range(0, y_hat.shape[0], limit):
            mse += torch.sum((y_hat[i: i + limit] - y[i: i + limit]) ** 2)
        return mse / y_hat.shape[0]
