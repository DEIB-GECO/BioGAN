import numpy as np 
import random
import torch
import torch.nn as nn
from utils.Parser import get_args
from utils.model_utils import *



class generator(nn.Module):

    def __init__(self, latent_dims,
                numerical_dims, 
                vocab_sizes,  
               generator_dims, stat1, stat2,
              negative_slope = 0.0, is_bn=False):
        super(generator, self).__init__()
    
        '''
        Parameters:
            
        '''
        parser = get_args()
        self.opt = parser.parse_args()
        print("self.opt.apply_norm_gen",self.opt.apply_norm_gen)
        self.latent_dims = latent_dims
        self.numerical_dims = len(numerical_dims)
        print('numerical_dims:', self.numerical_dims)
        self.vocab_sizes = vocab_sizes
        self.generator_dims = generator_dims
        self.negative_slope = negative_slope
        self.n_cat_vars = len(self.vocab_sizes)
        # concatenate noise vector + numerical covariates + embedded categorical covariates (e.g., tissue type)
        #self.input_dims = self.latent_dims+ self.numerical_dims +  self.categorical_embedded_dims * self.vocab_size
        self.categorical_embedded_dims = sum([int(vs**0.5)+1 for vs in self.vocab_sizes])
        self.input_dims = self.latent_dims + self.numerical_dims +  self.categorical_embedded_dims 
    
        self.generator = build_generator(self.input_dims, self.generator_dims[:-1], negative_slope=self.negative_slope, is_bn=is_bn)
        self.final_layer = nn.Linear(self.generator_dims[-2], self.generator_dims[-1])
        self.final_activation = nn.ReLU()
        # categorical embeddings
        self.categorical_embedding = categorical_embedding(self.vocab_sizes)
        self.stat1 = stat1 
        self.stat2 = stat2
        self.threshold = nn.Threshold(0,0)
  
    def forward(self, x, categorical_covariates):

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
        x_encoded = x

        for module in self.generator:

            x_encoded = module(x_encoded)
        
        #x_encoded = self.final_activation(self.final_layer(x_encoded))
        x_encoded  = self.final_layer(x_encoded)
        #x_encoded = self.threshold(x_encoded)
        # normalize the output
        
    
        if self.opt.apply_norm_gen:

            if self.opt.norm_type == 'min-max':
                
               x_encoded = normalize_output(x_encoded, max=self.stat1, min=self.stat2)
              

            elif self.opt.norm_type == 'standardize':
                x_encoded = normalize_output(x_encoded, mean=self.stat1, std=self.stat2)

        return x_encoded
    

    
class discriminator(nn.Module):

    def __init__(self, vector_dims,
                numerical_dims, 
                vocab_sizes,  
               discriminator_dims, negative_slope = 0.0, is_bn=False):
        super(discriminator, self).__init__()
        '''
        Take as input a gene expression sample and try to distinguish the true inputs
        '''

        self.vector_dims = vector_dims
        self.numerical_dims = len(numerical_dims)
        print('numerical_dims:', self.numerical_dims)
        self.vocab_sizes = vocab_sizes
        self.discriminator_dims = discriminator_dims
        self.negative_slope = negative_slope
        self.n_cat_vars = len(self.vocab_sizes)
        self.categorical_embedded_dims = sum([int(vs**0.5)+1 for vs in self.vocab_sizes])
        self.input_dims = self.vector_dims + self.numerical_dims +  self.categorical_embedded_dims 
        print(self.input_dims)

        self.discriminator = build_discriminator(self.input_dims, self.discriminator_dims[:-1], negative_slope = self.negative_slope, is_bn=is_bn)
        self.final_layer = nn.Linear(self.discriminator_dims[-2], self.discriminator_dims[-1])
        # categorical embeddings
        self.categorical_embedding = categorical_embedding(self.vocab_sizes)
        #self.embedding = nn.Embedding( self.vocab_size,  self.categorical_embedded_dims)
    

    def forward(self, x, categorical_covariates):

        embedded_cat_vars = []
        #cat_x = self.embeddinokg(categorical_covariates)
        for i, module in enumerate(self.categorical_embedding):
            cat_x = categorical_covariates[:,i]
            embedded_cat_vars.append(module(cat_x))
        if self.n_cat_vars == 1:
            embedded_cat_vars = embedded_cat_vars[0]
           
        else:
            embedded_cat_vars = torch.cat(embedded_cat_vars, dim=1)
        
    
        # for i in torch.unique(embedded_cat_vars, dim = 0):
        #     print(f"PIETRO: {i.sum()}")

        x = torch.cat((x,embedded_cat_vars), dim=1)
        x_encoded = x
    
        for module in self.discriminator:
            x_encoded = module(x_encoded)

        x_encoded = self.final_layer(x_encoded)
        #print(x_encoded)
        return x_encoded




def WGAN_GP_model(latent_dims,
                vector_dims,
                numerical_dims, 
                vocab_sizes, 
                generator_dims, 
                discriminator_dims, stat1, stat2,
                negative_slope = 0.0, is_bn= False):
    
    gen  = generator(latent_dims,
               numerical_dims, 
               vocab_sizes,  
               generator_dims, stat1, stat2,
               negative_slope, is_bn)
        
    disc = discriminator(vector_dims,
                                numerical_dims, 
                                vocab_sizes,  
                                discriminator_dims, 
                                negative_slope, is_bn)
        
    return gen, disc