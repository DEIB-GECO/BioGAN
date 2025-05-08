import numpy as np 
import random
import torch
import torch.nn as nn
from utils.Parser import get_args

# Set seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

def init_weights(m):
    # deafult is uniform_ normalization in pytorch ->  Kaiming He initialization
    # https://pytorch.org/docs/stable/generated/torch.nn.init.kaiming_uniform_.html for linear + relu

    if isinstance(m, nn.Linear):
        #torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def output_lsnorm(x, ls=1e6, epsilon=1e-6):
    # generate samples with a fixed library size
    sigmas = x.sum(dim=1, keepdim=True) + epsilon
    scales = x/sigmas
    x = scales * ls
    return x

def standardize(x, mean=None, std=None):
 
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    return (x - mean) / std

def min_max(x, max=None, min=None):
 
    if max is None:
        max = np.max(x, axis=0)
    if min is None:
        min = np.min(x, axis=0)
    return (x - min) / (max-min)

def normalize_output(x, mean=None, std=None, max=None, min=None):

    
    if mean is not None:
        x = x*std + mean
        x[x<0] = 0.0
        x = output_lsnorm(x)
        x = standardize(x, mean, std)

    if max is not None:
        x = x*(max-min) + min
        #x = min_max(x, max, min)
        x[x<0] = 0.0
        x = output_lsnorm(x)
        x = min_max(x, max, min)
        x[x<0] = 0.0
        
    #x = torch.round(x * 100) / 100
    
    return x


def build_linear_block(input_dims, output_dims, negative_slope = 0.0, is_bn= False):
    '''Paramters:
            -input_dims
            - output_dims
            - negative_slope: defeault 0.0 -> standard ReLU
            - is_bn -> batch normalization
    '''
    if is_bn:
        net = nn.Sequential(
            nn.Linear(input_dims, output_dims),
            nn.BatchNorm1d(output_dims),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
    else: 
        net = nn.Sequential(
            nn.Linear(input_dims, output_dims),
            nn.LeakyReLU(negative_slope=negative_slope))
            #nn.ReLU())
        

    return net

def build_generator(input_dims, generator_dims, negative_slope = 0.0, is_bn = False):

    generator = nn.ModuleList()
    for i in range(len(generator_dims)):
        if i == 0:
            print(input_dims)
            generator.append(build_linear_block(input_dims, generator_dims[i], negative_slope=negative_slope, is_bn= is_bn))
        else:
            generator.append(build_linear_block(generator_dims[i-1], generator_dims[i], negative_slope= negative_slope, is_bn= is_bn))
    return generator

def build_discriminator(input_dims, dicriminator_dims, negative_slope= 0.0, is_bn=False):

    dicriminator = nn.ModuleList()
    for i in range(len(dicriminator_dims)):
        if i == 0:
            dicriminator.append(build_linear_block(input_dims, dicriminator_dims[i], negative_slope= negative_slope, is_bn=is_bn))
        else:
            dicriminator.append(build_linear_block(dicriminator_dims[i-1], dicriminator_dims[i], negative_slope=negative_slope, is_bn=is_bn))
    return dicriminator


def build_encoder(input_dims, encoder_dims, negative_slope= 0.0, is_bn=False):

    encoder = nn.ModuleList()
    for i in range(len(encoder_dims)):
        if i == 0:
            encoder.append(build_linear_block(input_dims, encoder_dims[i], negative_slope= negative_slope, is_bn=is_bn))
        else:
            encoder.append(build_linear_block(encoder_dims[i-1], encoder_dims[i], negative_slope= negative_slope, is_bn=is_bn))
    return encoder

def build_decoder(latent_dims, decoder_dims, input_dims, negative_slope= 0.0, is_bn=False):

    decoder = nn.ModuleList()
    for i in range(len(decoder_dims)):
        if i == 0:
            decoder.append(build_linear_block(latent_dims, decoder_dims[i], negative_slope= negative_slope, is_bn=is_bn))
        else:
            decoder.append(build_linear_block(decoder_dims[i-1], decoder_dims[i], negative_slope= negative_slope, is_bn=is_bn))
    
    decoder.append(build_linear_block(decoder_dims[-1], input_dims))


    return decoder

def categorical_embedding(vocab_sizes):

    #n_cat_vars = len(vocab_sizes)
    embedder = nn.ModuleList()
    for vs in vocab_sizes:
        emdedding_dims = int(vs**0.5) +1 
        embedder.append(nn.Embedding(vs,  emdedding_dims))
    return embedder




