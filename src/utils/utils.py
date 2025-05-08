import os
import numpy as np
import pandas as pd
import warnings
import json
import torch 
import tqdm
from torch import nn
from joblib import Parallel, delayed
from datetime import datetime
import sys


def save_numpy(file, data):
    with open(file, 'wb') as f:
        np.save(f, data)

def create_folder(results_dire):    
    # create results dire
    if not os.path.exists(results_dire):
        os.makedirs(results_dire)
    
    else:
        warnings.warn('Folder already exists! Results will be overwritten!', category=Warning)
        
def save_dictionary(dictionary,filename):
    with open(filename + ".json", "w") as fp:
        json.dump(dictionary, fp)
        print("Done writing dict into .json file")

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)  
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)  

def save_checkpoint(results_path, epoch, filename, best_epoch, best_dice, best_epoch_loss,
                        best_val_loss, model, optimizer):
        path = os.path.join(results_path,"ckpt")
        os.makedirs(path, exist_ok=True)
        filename = filename + ".ckpt"
        try:
            torch.save({'epoch': epoch,
                        'best_epoch': best_epoch,
                        'best_epoch_dc_score': best_dice,
                        'best_epoch_loss': best_epoch_loss,
                        'best_val_loss': best_val_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        },os.path.join(path, filename))

        except Exception as e:
            print("An error occurred while saving the checkpoint:")
            print(e)


def arg_boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError("{} is not a valid boolean value".format(v))
    

def save_checkpoint_gan(results_path, gen, disc, epoch=None):
        path = os.path.join(results_path, "ckpt")
        os.makedirs(path, exist_ok=True)
        if isinstance(epoch, int):
            filename = 'model_'+str(epoch) + ".ckpt"
        else:
            filename = 'model_best' + ".ckpt"
        try:
            torch.save({'gen_state_dict': gen.state_dict(),
                        'disc_state_dict': disc.state_dict()},
                        os.path.join(path, filename))

        except Exception as e:
            print("An error occurred while saving the checkpoint:")
            print(e)


class EarlyStopper_gan:
    
    # we do not use loss but validation score
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        #self.min_validation_loss = float('inf')
        self.min_validation_loss = 0

    def early_stop(self, results_path, gen, disc, validation_loss, epoch):
        if validation_loss > self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.epoch = epoch
            self.counter = 0
            save_checkpoint_gan(results_path, gen, disc)
        elif validation_loss < (self.min_validation_loss + self.min_delta):
            self.counter += 1

            if self.counter >= self.patience:
                print('best_epoch:', self.epoch)
                return True
       
        return False





def open_file(dire):

    # g_dataset_to_use = dataset_to_use
    # g_task = task
    # g_dir_name = f"logs/log_{dataset_to_use}_{task}_{datetime.now().isoformat(sep='_', timespec='seconds')}"
    # # os.makedirs(g_dir_name)
    # os.makedirs(f"{g_dir_name}/images")
    # os.makedirs(f"{g_dir_name}/losses")
    # os.makedirs(f"{g_dir_name}/saved_models")
    f = open(
        f"{dire}/train_log.txt",
        "w",
    )
    return f


def reverse_normalization(data_real, data_gen, 
                        all_real, all_gen, stat1,
                        stat2, normalization, log2,rpm):

    print(stat1)
    if isinstance(stat1, torch.Tensor):
        stat1_cpu = stat1.cpu()  
        stat1 = stat1_cpu.numpy()
        
    if isinstance(stat2, torch.Tensor):
        stat2_cpu = stat2.cpu()  
        stat2 = stat2_cpu.numpy()    
    print('Reverting normalization')
    # revert normalization
    if normalization == 'standardize':
        print(pd.Series(data_real[:,0]).describe())
        data_real = data_real * stat2 + stat1
        print(pd.Series(data_real[:,0]).describe())
        data_gen = data_gen * stat2 + stat1
        all_real = all_real * stat2 + stat1
        all_gen = all_gen * stat2 + stat1

        data_real = np.where(data_real < 0, 0, data_real)
        data_gen = np.where(data_gen < 0, 0, data_gen)
        all_real = np.where(all_real < 0, 0, all_real)
        all_gen = np.where(all_gen < 0, 0, all_gen)

        if log2:
            data_real = 2**data_real - 1
            print(pd.Series(data_real[:,0]).describe())
            data_gen = 2**data_gen - 1
            all_real = 2**all_real -1
            all_gen = 2**all_gen - 1
            
            if not rpm:
                data_real = np.round(data_real)
                data_gen = np.round(data_gen)
                all_real = np.round(all_real)
                all_gen = np.round(all_gen)

        else:
            data_real = np.round(data_real)
            data_gen = np.round(data_gen)
            all_real = np.round(all_real)
            all_gen = np.round(all_gen)

    elif normalization == 'min-max':
        data_real = data_real * (stat1 - stat2) + stat2
        data_gen = data_gen * (stat1 - stat2) + stat2
        all_real = all_real * (stat1 - stat2) + stat2
        all_gen = all_gen * (stat1 - stat2) + stat2
        
        data_real = np.where(data_real < 0, 0, data_real)
        data_gen = np.where(data_gen < 0, 0, data_gen)
        all_real = np.where(all_real < 0, 0, all_real)
        all_gen = np.where(all_gen < 0, 0, all_gen)

        if log2:
            print('log2....')
            data_real = 2**data_real - 1
            data_gen = 2**data_gen - 1
            all_real = 2**all_real -1
            all_gen = 2**all_gen - 1
            
            if not rpm:
                
                print('Before Rounding.....')
                print("data real", data_real)
                data_real = np.round(data_real)
                data_gen = np.round(data_gen)
                all_real = np.round(all_real)
                all_gen = np.round(all_gen)
                print("data real", data_real)
                print("data_real", data_real)

        else:
            data_real = np.round(data_real)
            data_gen = np.round(data_gen)
            all_real = np.round(all_real)
            all_gen = np.round(all_gen)
        

    else:
     
        data_real = np.round(10**data_real)-1
        data_gen =  np.round(10**data_gen)-1
        all_real =  np.round(10**all_real) - 1
        all_gen =  np.round(10**all_gen) -1 
        # data_real = np.r(10**(10*data_real) -1)
        # print(data_real[:,0])
        # data_gen = np.trunc(10**(10*data_gen) -1)
        # print(data_gen[:,0])
        # all_real = np.trunc(10**(10*all_real) -1)
        # all_gen = np.trunc(10**(10*all_gen) -1)

    return data_real, data_gen, all_real, all_gen

