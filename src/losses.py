import numpy as np 
import random
import torch
import torch.nn as nn


def  wasserstein_loss(y_pred, y_true):
    #  distance function defined between probability distributions on a given metric space M
    return(torch.mean(y_pred * y_true))

def G_loss(fake_labels):
    # generator loss 
    # fake labels -> all fake samples are labeled as true samples 
    return wasserstein_loss(fake_labels, -torch.ones_like(fake_labels))
            
def D_loss(real_labels, fake_labels):

    loss_real = wasserstein_loss(-torch.ones_like(real_labels), real_labels)
    loss_fake = wasserstein_loss(torch.ones_like(fake_labels), fake_labels)
    total_loss = loss_real + loss_fake
    return total_loss, loss_real, loss_fake

def D_gan_loss(real_labels, fake_labels):

    loss_gan =  nn.BCEWithLogitsLoss()
    loss_real = loss_gan(real_labels, torch.ones_like(real_labels)) #(small error if real labels are close to 1)
    loss_fake = loss_gan(fake_labels, torch.zeros_like(fake_labels)) #(small error if fake labels are close to 0)
    total_loss = loss_real + loss_fake
    return total_loss, loss_real, loss_fake

# minimize error on real data
# maximize error on fake data 

def G_gan_loss(fake_labels):

    loss_gan =  nn.BCEWithLogitsLoss()
    loss_fake = loss_gan(fake_labels, torch.ones_like(fake_labels)) #(small error if fake labels are close to 1)
    return loss_fake