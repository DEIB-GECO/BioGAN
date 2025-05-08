# 3rd Party
import numpy as np
import random
#from dgl.nn import GraphConv, SAGEConv

import torch
from torch import nn
import time 
import torch_geometric
#from torch_geometric.data import Data
#from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SimpleConv, GraphConv, GatedGraphConv
from torch_sparse import SparseTensor
from torch_geometric.utils import is_torch_sparse_tensor, dense_to_sparse, to_torch_coo_tensor
from utils.utils import * 
from utils.model_coexpr_dorothea import *
from utils.pyg_utils import * 
# import functions
from model_utils import discriminator, categorical_embedding, discriminator_pcgan
# Set seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)



class GraphInputProcessor(nn.Module):
    def __init__(self, input_dim, output_dim, device, sparse=False):
        super(GraphInputProcessor, self).__init__()
        self.device = device
    
        #size of latent vector + cat embeddings
        self.mapping_layer = nn.Sequential(nn.Linear(input_dim, output_dim)).to(device)
        self.sparse = sparse
    
    
    def forward(self, z):
        """
        Prepares embeddings for graph decoding
            Parameters:
                z (Tensor): feature embeddings
                adj (Tensor): adjacency matrix
        
            Returns:
                b_z (Tensor): dense feature matrix, shape = (b_size*n_nodes, n_feats)
                b_adj (Tensor): batched adjacency matrix
                b_edge_weight (Sparse Tensor): sparse edge weights, shape = (n_edges)
        """

    

        mapped_z = self.mapping_layer(z)
        mapped_z =  mapped_z.unsqueeze(-1)
        #print('adj shape: ', adj.shape)
        #edge_index = dense_to_sparse(adj)
        #print('Non zero edges: ',  torch.nonzero(adj).size(0))
   
        

       

        return (mapped_z)


file =  '/data/my_data/graphs/collect_tri_ENS_graph_mapped.parquet'

aggr='mean'
class generator_gnn(nn.Module):
    def __init__(self, latent_dims, numerical_dims,  vocab_sizes, output_dim, 
                 decoder_dim, decoder_l, decoder_arch, device, genes_list,
                  learn_graph=True, masked=False, masking=0.5, random=False, batch=True, sparse=False, 
                  graph_file = file, thr=0.90):
        super(generator_gnn, self).__init__()
        decoder = nn.ModuleList([])
        #self.adj = build_graph('/home/mongardi/Synth_data/data/graphs/string_graph.parquet', genes_list, threshold=0.2) #.to(device)
        self.learn_graph = learn_graph
        if self.learn_graph:

            print('Learning graph')
            self.graph_learner = LearnedGraph_scratch(graph_file, genes_list,
             threshold=0.2, device=device, random=random, masked=masked)
           
        else:
            if random:
                self.adj = build_binary_random_graph(graph_file, genes_list)
            elif masked:
                self.adj = build_masked_graph(graph_file, genes_list, masking=masking)
            else:
                print("prior knowledge fix")
                self.adj = build_graph(graph_file, genes_list)
            print('Non zero edges: ',  torch.nonzero(self.adj).size(0))

            self.adj = abs(self.adj).to(device)
        
        self.latent_dims = latent_dims
        self.numerical_dims = len(numerical_dims)
        self.vocab_sizes = vocab_sizes
        self.categorical_embedded_dims = sum([int(vs**0.5)+1 for vs in self.vocab_sizes])
        self.final_activation = nn.ReLU()
        #self.negative_slope = negative_slope
        self.n_cat_vars = len(self.vocab_sizes)
        self.input_dims = self.latent_dims + self.numerical_dims +  self.categorical_embedded_dims 
        self.device = device
        self.decoder_l = decoder_l
        print('decoder_l:', decoder_l)
        self.batch = batch
        self.sparse = sparse


         # categorical embeddings
        self.categorical_embedding = categorical_embedding(self.vocab_sizes)
        self.decoder = nn.Sequential(*decoder)
        print('dev:', device)
        self.graph_processor = GraphInputProcessor(self.input_dims, output_dim, device)
        self.graph_model = H2GCN_dorothea(
        feat_dim=1,
        hidden_dim=8,
        out_dim=1,
        k=2, 
        use_relu=False,
         ).to(device)
        
    def forward(self, x, categorical_covariates):



        
        b_size = x.shape[0]
        adj_dict = {}
        adj_dict = {sample: self.adj for sample in range(b_size)}

        embedded_cat_vars = []
        #cat_x = self.embedding(categorical_covariates)
        for i, module in enumerate(self.categorical_embedding):
            cat_x = categorical_covariates[:,i]
            embedded_cat_vars.append(module(cat_x))
        if self.n_cat_vars == 1:
            embedded_cat_vars = embedded_cat_vars[0]
       
        else:
            embedded_cat_vars = torch.cat(embedded_cat_vars,dim=1)

        x = torch.cat((x,embedded_cat_vars), dim=1)
        z = x.clone()

        
        x_hat = []
     


        b_size = x.shape[0]




        for sample in range(b_size): 
                
            #z_sample = b_z[sample]
            #z_sample2 = b_z[sample]
             #print('z:', z_sample.shape)
            b_z = self.graph_processor(z[sample])
            b_z = self.graph_model(adj_dict, b_z, sample)
            #print("batch size", b_size)
            #print("b_z shape after layer:", b_z.shape)
            #print("b_z shape after squeeze:", b_z.squeeze().shape)
            #print(f"x_hat type before append: {type(x_hat)}") 
            x_hat.append(b_z.squeeze())

        x_hat =  torch.stack(x_hat)
 
        return x_hat


def WGAN_GP_model_gnn(latent_dims,
                vector_dims,
                numerical_dims, 
                vocab_sizes, 
                discriminator_dims, 
                output_dim, 
                decoder_dim, 
                device, genes, 
                learn_graph,
                coex_graph,
                two_tissues,
                masked, 
                masking, 
                random,
                decoder_l=3,
                decoder_arch='gcn',
                negative_slope = 0.0, is_bn= False, thr=0.90):
    
    # add info
    gen  = generator_gnn(latent_dims,
               numerical_dims, 
               vocab_sizes,  
                output_dim, 
                decoder_dim, 
                decoder_l, 
                decoder_arch, 
                device, genes, 
                learn_graph)
        
    disc = discriminator(vector_dims,
                                numerical_dims, 
                                vocab_sizes,  
                                discriminator_dims, 
                                negative_slope, is_bn)
        
    return gen, disc


def PC_WGAN_GP_model_gnn(latent_dims,
                vector_dims,
                numerical_dims, 
                vocab_sizes, 
                discriminator_dims, 
                output_dim, 
                decoder_dim, 
                device, genes, 
                decoder_l=3,
                decoder_arch='gcn',
                negative_slope = 0.0, is_bn= False):
    
    # add info
    gen  = generator_gnn(latent_dims,
               numerical_dims, 
               vocab_sizes,  
                output_dim, 
                decoder_dim, 
                decoder_l, 
                decoder_arch, 
                device, genes)
        
    disc = discriminator_pcgan(vector_dims,
                                numerical_dims, 
                                vocab_sizes,  
                                discriminator_dims, 
                                negative_slope, is_bn)
        
    return gen, disc

