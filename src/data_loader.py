import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from joblib import Parallel, delayed
from scipy import stats
import matplotlib.pyplot as plt

dict_mapper_tissue = None

def standardize(x, mean=None, std=None):
 
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    
    S = (x - mean) / std
    S[np.isnan(S)] = (x - mean)[np.isnan(S)]
    S[np.isinf(S)] = (x - mean)[np.isinf(S)]
    return S


def min_max(x, max=None, min=None):
 
    if max is None:
        max = np.max(x, axis=0)
    if min is None:
        min = np.min(x, axis=0)
    S = (x - min) / (max-min)
    S[np.isnan(S)] = (x - min)[np.isnan(S)]
    S[np.isinf(S)] = (x - min)[np.isinf(S)]
    return S

def seed_worker(worker_id):   
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def split_data(n_samples, train_rate=0.80, validation_rate=0.20, seed=42, shuffle=True):
    """
    Split data into train, validation, and test sets 
    :param sample_names: list of sample names
    :param train_rate: percentage of training samples
    :param validation_rate: percentage of validation samples
    :param seed: random seed
    :return: lists of train, validation, and test sample indices
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    idxs = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(idxs)
    t_tr = int(train_rate * (1-validation_rate) * n_samples)
    t_val = t_tr  + int(train_rate * validation_rate * n_samples)
    #t_t = t_val + int((1-train_rate)*n_samples)
    t_t = n_samples
    train_idxs = idxs[:t_tr]
    validation_idsx = idxs[t_tr:t_val]
    test_idxs = idxs[t_val:t_t]
    #print(train_idxs.shape[0] + validation_idsx.shape[0]+  test_idxs.shape[0])
    assert train_idxs.shape[0] + validation_idsx.shape[0]+  test_idxs.shape[0] == n_samples
    return train_idxs, validation_idsx, test_idxs



def load_data(data_dir, data_expr_file, data_info_file, graph_file=None, log2=False, log10=True, rpm=False, 
    remove_greedy_samples=True, variables=False,distinctness= False, less_distinctness= False, num_genes=3000):
    
    global dict_mapper_tissue

    # load gene expression file

    expr_df = pd.read_parquet(os.path.join(data_dir, data_expr_file))


    expr_values = expr_df.values.astype(float)
    
        
    print('initial dataset shape:',expr_df.shape)

    if log2:
        print('Log transforming data with log2')
        expr_values = np.log2(expr_values+1)

    if log10:
        print(expr_values[:,0])
        print('Log transforming data with log10, no other normalization is applied')
        #expr_values = np.log10(expr_values+1)
        expr_values = np.log10(expr_values+1)/10
        expr_values[np.isnan(expr_values)] = 0
        
        print(max(expr_values.max(axis=0)), min(expr_values.min(axis=0)))
        print(expr_values[:,0])
        # arcsinh-transformed data
        
    sample_names = expr_df.index.tolist()
    gene_names = expr_df.columns.tolist()
    #lsgene_names = [g.split('.')[0] for g in gene_names]
    print(expr_values.shape)
    print(f"The loaded file has {len(gene_names)} genes and {len(sample_names)} samples")

    gene_symbols = gene_names
    print('before:', expr_values.shape)


    print('after:', expr_values.shape)
    df_info =  pd.read_parquet(os.path.join(data_dir, data_info_file)) 


    discrete_conditions = df_info['SMTS']
    all_conditions = list(discrete_conditions.unique())
    dict_mapper_tissue = {k:i for i, k in enumerate(all_conditions)}

    #continuous_conditions = df_info['age_num'].values

    encoded_discrete_conditions = discrete_conditions.apply(lambda x: dict_mapper_tissue[x]).values
     
    return expr_values, gene_symbols, encoded_discrete_conditions, sample_names, discrete_conditions



def dataLoader_gtex(data_dir, data_expr_file, 
                    data_info_file, normalize= True, norm_type= 'standardize', log2=False, log10=True,
                    batch_size = 4, seed=42):
 
    # fix the generator for reproducible results
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)

    print('Loading data......')
    

    gene_expressions, gene_symbols, encoded_discrete_conditions, sample_names, discrete_conditions =  load_data(data_dir, data_expr_file, data_info_file,log2=log2, log10=log10)
   # idx = np.where((encoded_discrete_conditions[:, 0] != 8) & (encoded_discrete_conditions[:, 0] != 9) & (encoded_discrete_conditions[:, 0] != 14) & (encoded_discrete_conditions[:, 0] != 19) & (encoded_discrete_conditions[:, 0] != 21))
   # gene_expressions = gene_expressions[idx]
   # encoded_discrete_conditions = encoded_discrete_conditions[idx]
    #continuous_conditions = continuous_conditions[idx]
    
    print('Splitting data......')
  
    n_samples = gene_expressions.shape[0]
    
    train, validation, test = split_data(n_samples)
    print('train:', train.shape[0], 'validation:', validation.shape[0], 'test:', test.shape[0])
    
    expr_train, dc_train = gene_expressions[train], encoded_discrete_conditions[train]#, continuous_conditions[train]
    expr_validation, dc_validation = gene_expressions[validation], encoded_discrete_conditions[validation]#, continuous_conditions[validation]
    expr_test, dc_test= gene_expressions[test], encoded_discrete_conditions[test]#, continuous_conditions[test]
        
    # expr_train, dc_train = gene_expressions[train, :], encoded_discrete_conditions[train, :]#, continuous_conditions[train]
    # expr_validation, dc_validation = gene_expressions[validation, :], encoded_discrete_conditions[validation, :]#, continuous_conditions[validation]
    # expr_test, dc_test= gene_expressions[test, :], encoded_discrete_conditions[test, :]#, continuous_conditions[test]

    print(expr_train.shape, expr_validation.shape, expr_test.shape)
    
    n_genes = expr_train.shape[1]
    print("n_genes to generate", n_genes)       

    # standardize expression data 
    if normalize:
        print('Normalizing data......')
        if norm_type =='standardize':
            print('Normalizing standardize......')
            x_expr_mean = np.mean(expr_train, axis=0)
            x_expr_std = np.std(expr_train, axis=0)
            #print(np.where(x_expr_mean==0))
            #print(np.where(x_expr_std==0))
            idxs = np.where(x_expr_mean==0)[0]
            #print(idxs)
            print(pd.Series(np.concatenate((expr_train, expr_validation), axis=0)[:,0]).describe())
            expr_train = standardize(expr_train, mean=x_expr_mean, std=x_expr_std)
            # expr_train[np.isnan(expr_train)] = 0
            # expr_train[np.isinf(expr_train)] = 0
            expr_validation = standardize(expr_validation, mean=x_expr_mean, std=x_expr_std)

            expr_test = standardize(expr_test, mean=x_expr_mean, std=x_expr_std)

            # x_cc_mean = np.mean(cc_train)
            # x_cc_std = np.mean(cc_train)
            # cc_train = standardize(cc_train, mean=x_cc_mean, std=x_cc_std)
            # cc_validation = standardize(cc_validation, mean=x_cc_mean, std=x_cc_std)
            # cc_test = standardize(cc_test, mean=x_cc_mean, std=x_cc_std)
            print(pd.Series(np.concatenate((expr_train, expr_validation), axis=0)[:,0]).describe())
            
        if norm_type =='min-max':
            print('Normalizing min-max......')
            x_expr_max = np.max(expr_train, axis=0)
            x_expr_min = np.min(expr_train, axis=0)
            expr_train = min_max(expr_train, max=x_expr_max, min=x_expr_min)

            expr_validation = min_max(expr_validation, max=x_expr_max, min=x_expr_min)

            expr_test = min_max(expr_test, max=x_expr_max, min=x_expr_min)



            x_cc_max = np.max(cc_train)
            x_cc_min = np.min(cc_train)
            cc_train = min_max(cc_train, min=x_cc_max, max=x_cc_min)
            cc_validation = min_max(cc_validation, min=x_cc_max, max=x_cc_min)
            cc_test = min_max(cc_test, max=x_cc_max, min=x_cc_min)
            

    print('Building data loaders......')
    # build loaders
    print('n of cat vars:', dc_train)
    #print('n of cat vars:', dc_train.shape[1])
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(expr_train), 
                                            torch.from_numpy(dc_train).long()),
                                            batch_size=batch_size, shuffle=True,  worker_init_fn=seed_worker,
                                            generator=g)
    validation_loader = DataLoader(TensorDataset(torch.from_numpy(expr_validation),  
                                            torch.from_numpy(dc_validation).long()), 
                                            batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker,
                                            generator=g)
    test_loader =  DataLoader(TensorDataset(torch.from_numpy(expr_test), 
                                            torch.from_numpy(dc_test).long()),
                                            batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker,
                                            generator=g)
    
    vocab_sizes = [len(dict_mapper_tissue)]

    #[len(np.unique(encoded_discrete_conditions[:, i])) for i in range(encoded_discrete_conditions.shape[1])]
    print(vocab_sizes)
    print('Loading Completed!')
    if normalize and  norm_type=='standardize':
        return train_loader, validation_loader, test_loader, n_genes, gene_symbols, x_expr_mean , x_expr_std, vocab_sizes
    elif normalize and  norm_type=='min-max':
        return train_loader, validation_loader, test_loader, n_genes, gene_symbols, x_expr_max, x_expr_min, vocab_sizes
    else:
        return train_loader, validation_loader, test_loader, n_genes, gene_symbols, 0, 0, vocab_sizes