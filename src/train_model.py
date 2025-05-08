import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import pandas as pd
import numpy as np
from gnn.model_gnn import *
from data_loader import *

if __name__ == "__main__":
    data_dir = '../Multi_graph_synth/data' #da cambiare
    data_expr_file = 'subset4k.parquet'
    data_info_file = 'info_not_dummy_4k.parquet'

    train, val, test, n_genes, genes, stat1 , stat2, vocab_sizes = dataLoader_gtex(data_dir,
                                                            data_expr_file, data_info_file,
                                                           normalize= True, norm_type='standardize',
                                                            log2 = False,
                                                           log10 = False,
                                                          batch_size=64)

    lr_d = 1e-2
    lr_g = 1e-2
    gp_weight = 10
    optimizer = 'adam'
    patience = 10
    results_dir = '/results' #cambiare
    epochs = 400 

    model = WGAN_GP_gnn(input_dims= n_genes, 
                            latent_dims= 64,
                            vocab_sizes = vocab_sizes,  
                            generator_dims = [256, 256, n_genes],
                            discriminator_dims = [256, 256, 1], 
                            negative_slope = 0.01, is_bn= False,
                            lr_d = lr_d, lr_g = lr_g, 
                            optimizer= optimizer,
                            gp_weight = gp_weight,
                            p_aug=0.0, norm_scale=0.5, train=True, n_critic=5, freq_compute_test=100, freq_visualize_test=100,  
                            results_dire=results_dir, patience=patience)


    d_l = model.fit(train, val, test, epochs = epochs)