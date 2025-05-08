# 3rd Party
import numpy as np
import random
#from dgl.nn import GraphConv, SAGEConv
import pandas as pd
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
from utils.pyg_utils import * 
# import functions
from utils.model_utils import discriminator, categorical_embedding, discriminator_pcgan
# Set seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def build_graph_corr_synth(graph_file):
    print('Loading Graph...')
    df_graph = pd.read_parquet(graph_file)
    adj_matrix = torch.tensor(df_graph.values).to(torch.float32)
    # add self-loop
    adj_matrix.fill_diagonal_(1)
    return adj_matrix


class GraphInputProcessor(nn.Module):
    def __init__(self, input_dim, output_dim, device, sparse=False):
        super(GraphInputProcessor, self).__init__()
        self.device = device
    
        #size of latent vector + cat embeddings
        #self.mapping_layer = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU()).to(device)
        self.mapping_layer = nn.Sequential(nn.Linear(input_dim, output_dim)).to(device)
        self.sparse = sparse
    
    
    def forward(self, z, adj):
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

    
        b_size, n_nodes = z.shape
        mapped_z = self.mapping_layer(z)
        mapped_z =  mapped_z.unsqueeze(-1)

        edge_index = adj.nonzero().t()
        

        row, col = edge_index
        edge_weight = adj[row, col]

       

        return (mapped_z, edge_index, edge_weight)



file =  '/data/corr_matrices/graph_GO.parquet'

aggr='mean'
class generator_gnn(nn.Module):
    def __init__(self, latent_dims, numerical_dims,  vocab_sizes, output_dim, 
                 decoder_dim, decoder_l, decoder_arch, device, genes_list,
                  learn_graph=True, masked=False, masking=0.5, random=False, batch=True, sparse=False, 
                  graph_file = file, thr=0.90):
        super(generator_gnn, self).__init__()
        decoder = nn.ModuleList([])

        self.learn_graph = learn_graph
        self.adj = build_graph_corr_synth(graph_file)
        print('Non zero edges: ',  torch.nonzero(self.adj).size(0))
 
        self.adj = self.adj.to(device)



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
        if decoder_arch == "gcn":
            for i in range(decoder_l):

                if i ==0:
                    decoder.append(
                        GCNConv(1, decoder_dim, 
                            normalize=True,
                            bias=True, node_dim=1,))

                elif i == decoder_l - 1:
                    decoder.append(
                        GCNConv(decoder_dim_,
                            1,
                            normalize=True, 
                            bias=True, node_dim=1,
                        )
                    )
                    
                else:
                    decoder_dim_ = int(decoder_dim / 2)
                    decoder.append(
                
                        GCNConv(decoder_dim, decoder_dim_,
                            normalize=True,
                            bias=True, node_dim=1,
                        ))
                    
                    decoder_dim = decoder_dim_
        elif decoder_arch == "graphconv":
            
            for i in range(decoder_l):
       
                if i ==0:
                        decoder.append(
                            GraphConv(1, decoder_dim, 
                                bias=True, node_dim=1, aggr=aggr,))
                        decoder_dim_ = decoder_dim

                elif i == decoder_l - 1:
                    decoder.append(
                        GraphConv(decoder_dim_,
                            1, 
                            bias=True, node_dim=1, aggr=aggr,
                        )
                    )
                    
                else:
                    decoder_dim_ = int(decoder_dim_ / 2)
                    decoder.append(
                
                        GraphConv(decoder_dim, decoder_dim_,
                            bias=True, node_dim=1, aggr=aggr,
                        ))
                    decoder_dim = decoder_dim_

        elif decoder_arch == "gated":
            print('Gated')
            for i in range(decoder_l):
                if i ==0:
                        decoder.append(
                            GatedGraphConv(decoder_dim, num_layers=2,
                                bias=True, node_dim=1, aggr=aggr,))
                        decoder_dim_ = decoder_dim

                elif i == decoder_l - 1:
                    decoder.append(
                        GatedGraphConv(1, num_layers=2,
                            bias=True, node_dim=1, aggr=aggr,
                        )
                    )
                    
                else:
                    decoder_dim_ = int(decoder_dim_ / 2)
                    decoder.append(
                
                        GatedGraphConv(decoder_dim_, num_layers=2,
                            bias=True, node_dim=1, aggr=aggr,
                        ))
                    decoder_dim = decoder_dim_

        elif decoder_arch == "GIN":
            pass
        elif decoder_arch == "simple":

            for i in range(decoder_l):
                    decoder.append(
                        SimpleConv(aggr='sum', combine_root='sum',node_dim=1))
    
                    
        else:
            raise Exception("decoder can only be {het|gcn|sage}")

         # categorical embeddings
        self.categorical_embedding = categorical_embedding(self.vocab_sizes)
        self.decoder = nn.Sequential(*decoder) #si chiama decoder ma in realtà intendiamo il generatore  
        print('dev:', device)
        self.graph_processor = GraphInputProcessor(self.input_dims, output_dim, device)
        
    def forward(self, x, categorical_covariates):

        
        b_size = x.shape[0]
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
        if self.learn_graph:

            self.adj = self.graph_learner()
        graph_input = self.graph_processor(z, self.adj)
        b_z_orig, edge_index, edge_weight = graph_input 
        

        if self.sparse:
            n_nodes= b_z_orig.shape[1]
            adj = SparseTensor(row=edge_index[0], col=edge_index[1], 
                    value=edge_weight,
                    sparse_sizes=(n_nodes, n_nodes))
        else:
            adj = edge_index
   

        x_hat = []
        b_z = b_z_orig.clone()
        b_z2 = b_z_orig.clone()

        b_size = x.shape[0]

        if self.batch:
            

            for i, layer in enumerate(self.decoder):

                if self.sparse:
                    b_z = layer(b_z, adj.t(), edge_weight)
                else:
                    b_z = layer(b_z, adj, edge_weight)
                #print(b_z.shape)
                if i != self.decoder_l - 1:
                    #b_z = b_z.tanh()
                    #b_z = torch.sigmoid(b_z)
                    b_z = torch.nn.functional.relu(b_z)
                # edge_weight.to(self.device)

            return b_z.squeeze()
    
        else:
        # much slower   
            # #print(x_hat.shape)
            # return b_z.squeeze()

            for sample in range(b_size): 

                z_sample = b_z[sample]
                z_sample2 = b_z[sample]
                #print('z:', z_sample.shape)
                for layer in self.decoder:

                    #z_sample = layer(adj.to(self.device), z_sample, edge_weight=edge_weight.to(self.device))
                    z_sample = layer(z_sample, adj, edge_weight)

                    
                x_hat.append(z_sample)

            x_hat =  torch.stack(x_hat).squeeze(-1)
            x_hat = self.final_activation(x_hat)
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


