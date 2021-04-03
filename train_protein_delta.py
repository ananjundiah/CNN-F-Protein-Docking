from __future__ import print_function
import os
import logging
import numpy as np
import random
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import shutil
from shutil import copyfile
from datetime import datetime
from tensorboardX import SummaryWriter
from cnnf.model_cifar import WideResNet
from cnnf.model_protein_contact import CNNF

from utils import *
from advertorch.attacks import GradientSignAttack, LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval

def train_adv(args, model, device, train_loader, optimizer, scheduler, epoch,
          cycles, mse_parameter=1.0, clean_parameter=1.0, clean='supclean'):

    model.train()

    train_loss = 0.0
    train_pred_loss = 0.0
    train_recon_loss = 0.0
    train_final_recon_loss = 0.0
    train_final_pred_loss = 0.0

    model.reset()

    sig = nn.Sigmoid()
    rel = nn.ReLU()

    for batch_idx, sample in enumerate(train_loader):
            
        optimizer.zero_grad()

        # print('Batch index: ' + str(batch_idx))
        # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
        # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))

        #Get features
        # print('Loading maps...')

        contact_map = sample['contact_map'].to(device)
        score = sample['score'].to(device)
        contact_ref_map = sample['contact_ref_map'].to(device)
        score_ref = sample['score_ref'].to(device)

        # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
        # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))

        model.reset()
        # print('Concatenating...')
        maps_all = torch.cat((contact_ref_map, contact_map), 0)

        # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
        # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))
              
        #Reset the model latent variables
        model.reset()

        #Run first pass through network
        # print('First pass...')
        logits, orig_feature_all, block1_all, block2_all = model(maps_all, first=True, inter=True)

        # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
        # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))

        ff_prev = orig_feature_all

        # print('Splitting...')

        # find the original feature of clean images
        orig_feature, _ = torch.split(orig_feature_all, contact_ref_map.size(0))
        block1_clean, _ = torch.split(block1_all, contact_ref_map.size(0))
        block2_clean, _ = torch.split(block2_all, contact_ref_map.size(0))
        logits_clean, logits_adv = torch.split(logits, contact_ref_map.size(0))

        orig_feature_cm = orig_feature[:, 0, :, :]
        cm_idx = orig_feature_cm != 0

        # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
        # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))

        # print('Calculating first pass loss...')

        #Calculate the prediction loss for first pass
        if not ('no' in clean):
            pred_loss = (clean_parameter * (F.mse_loss(sig(logits_clean[:,0]), score_ref[:,0]) + F.mse_loss(rel(logits_clean[:,1]), score_ref[:,1])) + (F.mse_loss(sig(logits_adv[:,0]), score_ref[:,0]) + F.mse_loss(rel(logits_adv[:,1]), score_ref[:,1]))) / (4*(cycles+1))
            recon_loss = 0
            loss = (clean_parameter * (F.mse_loss(sig(logits_clean[:,0]), score_ref[:,0]) + F.mse_loss(rel(logits_clean[:,1]), score_ref[:,1])) + (F.mse_loss(sig(logits_adv[:,0]), score_ref[:,0]) + F.mse_loss(rel(logits_adv[:,1]), score_ref[:,1]))) / (4*(cycles+1))
        else:        
            pred_loss = (F.mse_loss(sig(logits_adv[:,0]), score_ref[:,0]) + F.mse_loss(rel(logits_adv[:,1]), score_ref[:,1])) / (2*(cycles+1))
            recon_loss = 0
            loss = (F.mse_loss(sig(logits_adv[:,0]), score_ref[:,0]) + F.mse_loss(rel(logits_adv[:,1]), score_ref[:,1])) / (2*(cycles+1))

        # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
        # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))

        for i_cycle in range(cycles):

            #Run backwards pass and split
            # print('Recon for i cycle ' + str(i_cycle))
            recon, block1_recon, block2_recon = model(logits, step='backward', inter_recon=True)
            recon_clean, recon_adv = torch.split(recon, contact_ref_map.size(0))
            recon_block1_clean, recon_block1_adv = torch.split(block1_recon, contact_ref_map.size(0))
            recon_block2_clean, recon_block2_adv = torch.split(block2_recon, contact_ref_map.size(0))

            # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
            # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))

            # print('Calculating recon loss...')
            #Calculate the reconstruction loss

            recon_loss += (F.mse_loss(recon_adv[:,0,:,:][cm_idx], orig_feature_cm[cm_idx]) + F.mse_loss(recon_block1_adv, block1_clean) + F.mse_loss(
                recon_block2_adv, block2_clean)) * mse_parameter / (3 * cycles)
            loss += (F.mse_loss(recon_adv[:,0,:,:][cm_idx], orig_feature_cm[cm_idx]) + F.mse_loss(recon_block1_adv, block1_clean) + F.mse_loss(
                recon_block2_adv, block2_clean)) * mse_parameter / (3 * cycles)

            # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
            # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))

            # print('Update previous')
            #Update the previous input
            ff_current = ff_prev + args.res_parameter * (recon - ff_prev)
            ff_current[:,1:,:,:] = ff_prev[:,1:,:,:]

            # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
            # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))

            # print('Run forward for i cycle ' + str(i_cycle))
            #Run forwards pass and split
            logits = model(ff_current, first=False)
            ff_prev = ff_current
            logits_clean, logits_adv = torch.split(logits, contact_ref_map.size(0))

            # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
            # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))

            # print('Calculating prediction loss...')
            #Calculate the prediction loss
            if not ('no' in clean):
                pred_loss += (clean_parameter * (F.mse_loss(sig(logits_clean[:, 0]), score_ref[:, 0]) + F.mse_loss(rel(logits_clean[:, 1]),score_ref[:, 1])) + (F.mse_loss(sig(logits_adv[:, 0]), score_ref[:, 0]) + F.mse_loss(rel(logits_adv[:, 1]), score_ref[:, 1]))) / (4 * (cycles + 1))
                loss += (clean_parameter * (F.mse_loss(sig(logits_clean[:, 0]), score_ref[:, 0]) + F.mse_loss(rel(logits_clean[:, 1]),score_ref[:, 1])) + (F.mse_loss(sig(logits_adv[:, 0]), score_ref[:, 0]) + F.mse_loss(rel(logits_adv[:, 1]), score_ref[:, 1]))) / (4 * (cycles + 1))
            else:
                # loss += F.cross_entropy(logits_adv, targets) / (cycles+1)
                pred_loss += (F.mse_loss(sig(logits_adv[:, 0]), score_ref[:, 0]) + F.mse_loss(rel(logits_adv[:, 1]), score_ref[:, 1])) / (2 * (cycles + 1))
                loss += (F.mse_loss(sig(logits_adv[:, 0]), score_ref[:, 0]) + F.mse_loss(rel(logits_adv[:, 1]), score_ref[:, 1])) / (2 * (cycles + 1))

            # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
            # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))

        # print('Calculating final recon and pred loss...')
        #Calculate the final prediction and reconstruction loss
        _, ff_adv = torch.split(ff_prev, contact_ref_map.size(0))
        final_recon_loss = F.mse_loss(ff_adv[:,0,:,:][cm_idx], orig_feature_cm[cm_idx])
        final_pred_loss = (F.mse_loss(sig(logits_adv[:, 0]), score_ref[:, 0]) + F.mse_loss(rel(logits_adv[:, 1]), score_ref[:, 1]))/2

        # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
        # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))
        #
        # print('Gradient step')
        #Gradient step
        loss.backward()
        if (args.grad_clip):
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
        # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))
        #
        # print('Add to total losses')
        #Add to total across batches for epoch
        train_loss += loss.item()
        train_pred_loss += pred_loss.item()
        train_recon_loss += recon_loss.item()
        train_final_recon_loss += final_recon_loss.item()
        train_final_pred_loss += final_pred_loss.item()

        # print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
        # print('Memory reserved: ' + str(torch.cuda.memory_reserved()))

        #Print batch loss for particular interval
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPrediction Loss: {:.6f}\tRecon Loss: {:.6f}\tFinal Pred Loss: {:.6f}\tFinal Recon Loss: {:.6f}'.format(
                epoch, batch_idx * contact_ref_map.size(0), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), pred_loss.item(),recon_loss.item(),final_pred_loss.item(),final_recon_loss.item()))

    #Get average loss
    train_loss /= len(train_loader)
    train_pred_loss /= len(train_loader)
    train_recon_loss /= len(train_loader)
    train_final_pred_loss /= len(train_loader)
    train_final_recon_loss /= len(train_loader)

    return train_loss, train_pred_loss, train_recon_loss, train_final_pred_loss,train_final_recon_loss

def train_delta(args, model, device, train_loader, optimizer, scheduler, epoch,
              cycles, mse_parameter=1.0):
    model.train()

    train_pred_loss = 0.0
    train_recon_loss = 0.0
    train_final_recon_loss = 0.0
    train_final_pred_loss = 0.0
    cm_sum_list = []
    model.reset()

    for batch_idx, sample in enumerate(train_loader):

        optimizer.zero_grad()

        contact_map = sample['contact_map'].to(device)
        score = sample['score'].to(device)
        contact_ref_map = sample['contact_ref_map'].to(device)
        score_ref = sample['score_ref'].to(device)

        score_delta = score_ref - score

        # Reset the model latent variables
        model.reset()

        #Run first forward pass
        logits_list = []
        logits = model(contact_map, first=True, inter=False)
        ff_prev = contact_map
        logits_list.append(logits)

        #Find the index of non-zero entries
        orig_feature_cm = contact_ref_map[:, 0, :, :]
        cm_idx = orig_feature_cm != 0

        # Calculate the prediction loss for first pass

        pred_loss = F.mse_loss(logits, score_delta)/(cycles + 1)
        recon_loss = 0
        loss = F.mse_loss(logits, score_delta)/(cycles + 1)

        for i_cycle in range(cycles):

            # Run backwards pass

            recon = model(logits, step='backward', inter_recon=False)
            recon_loss += F.mse_loss(recon[:, 0, :, :][cm_idx], orig_feature_cm[cm_idx] - ff_prev[:, 0, :, :][cm_idx]) * mse_parameter/cycles
            loss += F.mse_loss(recon[:, 0, :, :][cm_idx], orig_feature_cm[cm_idx] - ff_prev[:, 0, :, :][cm_idx]) * mse_parameter/cycles

            # Update the previous input
            ff_current = ff_prev + args.res_parameter * recon
            ff_current[:, 1:, :, :] = ff_prev[:, 1:, :, :]

            # Run forwards pass and split
            logits = model(ff_current, first=False)
            ff_prev = ff_current

            # Calculate the prediction loss
            score_delta = score_ref - (score + args.pred_res_parameter * sum(logits_list))
            pred_loss += F.mse_loss(logits, score_delta)/(cycles + 1)
            loss += F.mse_loss(logits, score_delta)/(cycles + 1)
            logits_list.append(logits)

        # Calculate the final prediction and reconstruction loss
        final_recon_loss = F.mse_loss(ff_prev[:, 0, :, :][cm_idx], orig_feature_cm[cm_idx])
        final_score = score + args.pred_res_parameter * sum(logits_list)
        final_score[:, 0] = torch.clamp(final_score[:, 0], 0, 1)
        final_score[:, 1] = torch.clamp(final_score[:, 1], 0)
        final_pred_loss = F.mse_loss(final_score, score_ref)

        # Gradient step
        loss.backward()
        if (args.grad_clip):
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        # Add to total across batches for epoch
        cm_sum = (torch.sum(cm_idx).item()) / 100000
        cm_sum_list.append(cm_sum)
        train_pred_loss += pred_loss.item() * contact_map.size(0)
        train_recon_loss += recon_loss.item() * cm_sum
        train_final_recon_loss += final_recon_loss.item() * cm_sum
        train_final_pred_loss += final_pred_loss.item() * contact_map.size(0)

        # Print batch loss for particular interval
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPrediction Loss: {:.6f}\tRecon Loss: {:.6f}\tFinal Pred Loss: {:.6f}\tFinal Recon Loss: {:.6f}'.format(
                    epoch, batch_idx * contact_ref_map.size(0), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), pred_loss.item(), recon_loss.item(),
                    final_pred_loss.item(), final_recon_loss.item()))

    # Get average loss
    train_pred_loss /= len(train_loader.dataset)
    train_recon_loss /= sum(cm_sum_list)
    train_final_pred_loss /= len(train_loader.dataset)
    train_final_recon_loss /= sum(cm_sum_list)
    train_loss = train_pred_loss + train_recon_loss

    return train_loss, train_pred_loss, train_recon_loss, train_final_pred_loss, train_final_recon_loss

def test(args, model, device, test_loader, cycles, epoch, mse_parameter = 1.0, clean_parameter = 1.0):
    model.eval()
    test_loss = 0.0
    test_pred_loss = 0.0
    test_recon_loss = 0.0
    test_final_pred_loss = 0.0
    test_final_recon_loss = 0.0
    sig = nn.Sigmoid()
    rel = nn.ReLU()
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            contact_map = sample['contact_map'].to(device)
            score = sample['score'].to(device)
            contact_ref_map = sample['contact_ref_map'].to(device)
            score_ref = sample['score_ref'].to(device)

            # Run first pass through network
            model.reset()
            maps_all = torch.cat((contact_ref_map, contact_map), 0)
            output, orig_feature_all, block1_all, block2_all = model(maps_all, first=True, inter=True)
            ff_prev = orig_feature_all

            #Split features
            orig_feature, _ = torch.split(orig_feature_all, contact_ref_map.size(0))
            logits_clean, logits_adv = torch.split(output, contact_ref_map.size(0))
            block1_clean, _ = torch.split(block1_all, contact_ref_map.size(0))
            block2_clean, _ = torch.split(block2_all, contact_ref_map.size(0))

            #Get just the contact map and indices for non-zero entries
            orig_feature_cm = orig_feature[:, 0, :, :]
            cm_idx = orig_feature_cm != 0

            #Loss for first pass
            pred_loss = (clean_parameter * (F.mse_loss(sig(logits_clean[:,0]), score_ref[:,0]) + F.mse_loss(rel(logits_clean[:,1]), score_ref[:,1])) + (F.mse_loss(sig(logits_adv[:,0]), score_ref[:,0]) + F.mse_loss(rel(logits_adv[:,1]), score_ref[:,1]))) / (4*(cycles+1))
            recon_loss = 0.0
            loss = 0.0
            loss += pred_loss

            for i_cycle in range(cycles):

                #Run backwards pass and split
                recon, block1_recon, block2_recon = model(output, step='backward', inter_recon=True)
                recon_clean, recon_adv = torch.split(recon, contact_ref_map.size(0))
                recon_block1_clean, recon_block1_adv = torch.split(block1_recon, contact_ref_map.size(0))
                recon_block2_clean, recon_block2_adv = torch.split(block2_recon, contact_ref_map.size(0))

                #Calculate reconstruction loss
                rloss = (F.mse_loss(recon_adv[:,0,:,:][cm_idx], orig_feature_cm[cm_idx]) + F.mse_loss(recon_block1_adv, block1_clean) + F.mse_loss(
                    recon_block2_adv, block2_clean)) * mse_parameter / (3 * cycles)
                recon_loss += rloss
                loss += rloss

                #Update input features
                ff_current = ff_prev + args.res_parameter * (recon - ff_prev)
                ff_current[:, 1:, :, :] = ff_prev[:, 1:, :, :]

                #Run forwards pass and split
                output = model(ff_current, first=False)
                logits_clean, logits_adv = torch.split(output, contact_ref_map.size(0))
                ff_prev = ff_current

                #Calculate prediction loss
                ploss = (clean_parameter * (
                            F.mse_loss(sig(logits_clean[:, 0]), score_ref[:, 0]) + F.mse_loss(rel(logits_clean[:, 1]),
                                                                                              score_ref[:, 1])) + (
                                          F.mse_loss(sig(logits_adv[:, 0]), score_ref[:, 0]) + F.mse_loss(
                                      rel(logits_adv[:, 1]), score_ref[:, 1]))) / (4 * (cycles + 1))
                pred_loss += ploss
                loss += ploss

            #Get final prediction and reconstruction losses
            _, ff_adv = torch.split(ff_prev, contact_ref_map.size(0))
            final_recon_loss = F.mse_loss(ff_adv[:,0,:,:][cm_idx], orig_feature_cm[cm_idx])
            final_pred_loss = (F.mse_loss(sig(logits_adv[:, 0]), score_ref[:, 0]) + F.mse_loss(rel(logits_adv[:, 1]), score_ref[:, 1]))/2

            #Multiply by size of batch, since there are uneven sizes
            test_loss += loss.item() * contact_ref_map.size(0)
            test_pred_loss += pred_loss.item() * contact_ref_map.size(0)
            test_recon_loss += recon_loss.item() * contact_ref_map.size(0)
            test_final_recon_loss += final_recon_loss.item() * contact_ref_map.size(0)
            test_final_pred_loss += final_pred_loss.item() * contact_ref_map.size(0)

    #Get average loss by dividing by sample number instead of batch number
    test_loss /= len(test_loader.dataset)
    test_pred_loss /= len(test_loader.dataset)
    test_recon_loss /= len(test_loader.dataset)
    test_final_recon_loss /= len(test_loader.dataset)
    test_final_pred_loss /= len(test_loader.dataset)

    print('\nTest set: Whole test loss: {:.4f}, Prediction loss: {:.4f}, Recon loss: {:.4f}, Final pred loss: {:.4f}, Final recon loss: {:.4f}\n'.format(
        test_loss, test_pred_loss, test_recon_loss, test_final_pred_loss, test_final_recon_loss))

    return test_loss, test_pred_loss, test_recon_loss, test_final_pred_loss,test_final_recon_loss

def test_delta(args, model, device, test_loader, cycles, epoch, mse_parameter = 1.0):
    model.eval()
    test_pred_loss = 0.0
    test_recon_loss = 0.0
    test_final_pred_loss = 0.0
    test_final_recon_loss = 0.0
    cm_sum_list = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            contact_map = sample['contact_map'].to(device)
            score = sample['score'].to(device)
            contact_ref_map = sample['contact_ref_map'].to(device)
            score_ref = sample['score_ref'].to(device)

            score_delta = score_ref - score

            # Reset the model latent variables
            model.reset()

            # Run first forward pass
            logits_list = []
            logits = model(contact_map, first=True, inter=False)
            ff_prev = contact_map
            logits_list.append(logits)

            # Find the index of non-zero entries
            orig_feature_cm = contact_ref_map[:, 0, :, :]
            cm_idx = orig_feature_cm != 0

            # Calculate the prediction loss for first pass

            pred_loss = F.mse_loss(logits, score_delta) / (cycles + 1)
            recon_loss = 0
            loss = F.mse_loss(logits, score_delta) / (cycles + 1)

            for i_cycle in range(cycles):
                # Run backwards pass and split

                recon = model(logits, step='backward', inter_recon=False)
                recon_loss += F.mse_loss(recon[:, 0, :, :][cm_idx],
                                         orig_feature_cm[cm_idx] - ff_prev[:, 0, :, :][cm_idx]) * mse_parameter / cycles
                loss += F.mse_loss(recon[:, 0, :, :][cm_idx],
                                   orig_feature_cm[cm_idx] - ff_prev[:, 0, :, :][cm_idx]) * mse_parameter / cycles

                # Update the previous input
                ff_current = ff_prev + args.res_parameter * recon
                ff_current[:, 1:, :, :] = ff_prev[:, 1:, :, :]

                # Run forwards pass and split
                logits = model(ff_current, first=False)
                ff_prev = ff_current

                # Calculate the prediction loss
                score_delta = score_ref - (score + args.pred_res_parameter * sum(logits_list))
                pred_loss += F.mse_loss(logits, score_delta) / (cycles + 1)
                loss += F.mse_loss(logits, score_delta) / (cycles + 1)
                logits_list.append(logits)

            # Calculate the final prediction and reconstruction loss
            final_recon_loss = F.mse_loss(ff_prev[:, 0, :, :][cm_idx], orig_feature_cm[cm_idx])
            final_score = score + args.pred_res_parameter * sum(logits_list)
            final_score[:, 0] = torch.clamp(final_score[:, 0], 0, 1)
            final_score[:, 1] = torch.clamp(final_score[:, 1], 0)
            final_pred_loss = F.mse_loss(final_score, score_ref)

            #Multiply by size of batch, since there are uneven sizes
            cm_sum = (torch.sum(cm_idx).item())/100000
            cm_sum_list.append(cm_sum)
            test_pred_loss += pred_loss.item() * contact_map.size(0)
            test_recon_loss += recon_loss.item() * cm_sum
            test_final_recon_loss += final_recon_loss.item() * cm_sum
            test_final_pred_loss += final_pred_loss.item() * contact_map.size(0)

    #Get average loss by dividing by sample number instead of batch number
    test_pred_loss /= len(test_loader.dataset)
    test_recon_loss /= sum(cm_sum_list)
    test_final_recon_loss /= sum(cm_sum_list)
    test_final_pred_loss /= len(test_loader.dataset)
    test_loss = test_pred_loss + test_recon_loss

    print('\nTest set: Whole test loss: {:.4f}, Prediction loss: {:.4f}, Recon loss: {:.4f}, Final pred loss: {:.4f}, Final recon loss: {:.4f}\n'.format(
        test_loss, test_pred_loss, test_recon_loss, test_final_pred_loss, test_final_recon_loss))

    return test_loss, test_pred_loss, test_recon_loss, test_final_pred_loss,test_final_recon_loss

def test_pgd(args, model, device, test_loader, epsilon=0.063):
    
    model.eval()
    model.reset()

    for batch_idx, (data, target) in enumerate(test_loader):
        contact_map = sample['contact_map'].to(device)
        score = sample['score'].to(device)
        contact_ref_map = sample['contact_ref_map'].to(device)
        score_ref = sample['score_ref'].to(device)
        model.reset()

        output = model.run_cycles(adv_images)

        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()

    acc = correct / len(test_loader.dataset)
    print('PGD attack Acc {:.3f}'.format(100. * acc))

    return acc

class ContactMapDataset(Dataset):
    """Contact map dataset."""
    def __init__(self, path, score_idx = [1,2]):
        """
        Args:
            path (string): Path to the data.
            score_idx: which elements of the 4 scores to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path = path
        self.id = np.sort(os.listdir(self.path))
        self.score_idx = score_idx
        self.id = np.array([i.replace('_Reference_ContactMap.npz', '') for i in self.id if '_Reference_ContactMap.npz' in i])
    def __len__(self):
        return len(self.id)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.id[idx]
        contact_map = torch.from_numpy(np.load(os.path.join(self.path,name + '_ContactMap.npz'))['arr_0']).float()
        score = torch.from_numpy(np.loadtxt(os.path.join(self.path,name + '_Scores.csv'))).float()
        score = score[self.score_idx]
        contact_ref_map = torch.from_numpy(np.load(os.path.join(self.path,name + '_Reference_ContactMap.npz'))['arr_0']).float()
        score_ref = torch.from_numpy(np.loadtxt(os.path.join(self.path,name + '_Reference_Scores.csv'))).float()
        score_ref = score_ref[self.score_idx]
        sample = {'contact_map': contact_map, 'score': score, 'contact_ref_map': contact_ref_map, 'score_ref': score_ref, 'id': name}
        return sample


def main():
    parser = argparse.ArgumentParser(description='CNNF training')
    # optimization parameters
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128 for CIFAR, 64 for MNIST)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.15, metavar='LR',
                        help='learning rate (default: 0.05 for SGD)')
    parser.add_argument('--power', type=float, default=0.9, metavar='LR',
                        help='learning rate for poly scheduling')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
    parser.add_argument('--grad-clip', action='store_true', default=False,
                        help='enable gradient clipping')
    # parser.add_argument('--dataset', choices=['cifar10', 'fashion'],
    #                     default='cifar10', help='the dataset for training the model')
    parser.add_argument('--path_train', default='.', help='the path for training data')
    parser.add_argument('--path_test', default='.', help='the path for test data')
    parser.add_argument('--schedule', choices=['poly', 'cos'],
                        default='poly', help='scheduling for learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status')
    
    # adversarial training parameters
    parser.add_argument('--eps', type=float, default=0.063,
                        help='Perturbation magnitude for adv training')
    parser.add_argument('--eps-iter', type=float, default=0.02,
                        help='attack step size')
    parser.add_argument('--nb_iter', type=int, default=7,
                        help='number of steps in pgd attack')
    parser.add_argument('--clean', choices=['no', 'supclean'],
                        default='supclean', help='whether to use clean data in adv training')
    
    # hyper-parameters
    parser.add_argument('--mse-parameter', type=float, default=1.0,
                        help='weight of the reconstruction loss')
    parser.add_argument('--clean-parameter', type=float, default=1.0,
                        help='weight of the clean Xentropy loss')
    parser.add_argument('--res-parameter', type=float, default=0.1,
                        help='step size for residuals')
    parser.add_argument('--pred-res-parameter', type=float, default=1.0,
                        help='step size for prediction residuals')
    
    # model parameters
    parser.add_argument('--layers', default=40, type=int, help='total number of layers for WRN')
    parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor for WRN')
    parser.add_argument('--droprate', default=0.0, type=float, help='Dropout probability')
    parser.add_argument('--ind', type=int, default=2,
                        help='index of the intermediate layer to reconstruct to')
    parser.add_argument('--max-cycles', type=int, default=2,
                        help='the maximum cycles that the CNN-F uses')
    parser.add_argument('--save-model', default=None,
                        help='Name for Saving the current Model')
    parser.add_argument('--model-dir', default=None,
                        help='Directory for Saving the current Model')

    
    args = parser.parse_args()
 
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    seed_torch(args.seed)

    Tensor_writer = SummaryWriter(os.path.join(args.model_dir, args.save_model))

    #Load the training and test data
    print('Loading training data')
    train_data = ContactMapDataset(path = args.path_train, score_idx=[1,2])
    train_loader = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      shuffle=True, pin_memory=False, drop_last = False , num_workers=4)
    print('Loading test data')
    test_data = ContactMapDataset(path=args.path_test, score_idx=[1, 2])
    test_loader = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size,
      shuffle=True, pin_memory=False, drop_last = False , num_workers=4)

    #Create network
    num_metrics = 2
    model = CNNF(num_metrics, ind=args.ind, cycles=args.max_cycles, res_param=args.res_parameter).to(device)

    optimizer = torch.optim.SGD(
          model.parameters(),
          args.lr,
          momentum=args.momentum,
          weight_decay=args.wd)
            
    if(args.schedule == 'cos'):        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
          optimizer, lr_lambda=lambda step: get_lr(step, args.epochs * len(train_loader), 1.0, 1e-5))
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
          optimizer, lr_lambda=lambda step: lr_poly(1.0, step, args.epochs * len(train_loader), args.power))

    # Begin training
    best_loss = 0

    for epoch in range(args.epochs):
        print('Run train epoch ' + str(epoch))
        train_loss, train_pred_loss, train_recon_loss, train_final_pred_loss, train_final_recon_loss = train_delta(args, model, device, train_loader, optimizer, scheduler, epoch,
          cycles=args.max_cycles, mse_parameter=args.mse_parameter)
        print('Run test epoch ' + str(epoch))
        test_loss, test_pred_loss, test_recon_loss, test_final_pred_loss, test_final_recon_loss = test_delta(args, model, device, test_loader, cycles=args.max_cycles, epoch=epoch, mse_parameter=args.mse_parameter)
        
        Tensor_writer.add_scalars('loss', {'train': train_loss}, epoch)
        Tensor_writer.add_scalars('pred loss', {'train': train_pred_loss}, epoch)
        Tensor_writer.add_scalars('recon loss', {'train': train_recon_loss}, epoch)
        Tensor_writer.add_scalars('final pred loss', {'train': train_final_pred_loss}, epoch)
        Tensor_writer.add_scalars('final recon loss', {'train': train_final_recon_loss}, epoch)

        Tensor_writer.add_scalars('loss', {'test': test_loss}, epoch)
        Tensor_writer.add_scalars('pred loss', {'test': test_pred_loss}, epoch)
        Tensor_writer.add_scalars('recon loss', {'test': test_recon_loss}, epoch)
        Tensor_writer.add_scalars('final pred loss', {'test': test_final_pred_loss}, epoch)
        Tensor_writer.add_scalars('final recon loss', {'test': test_final_recon_loss}, epoch)

        # Save the model with the best accuracy
        if ((test_final_recon_loss + test_final_pred_loss) < best_loss and args.save_model is not None) or (best_loss == 0 and args.save_model is not None):
            best_loss = test_final_recon_loss + test_final_pred_loss
            print('Best loss = ' + str(best_loss))
            experiment_fn = args.save_model
            torch.save(model.state_dict(),
                       args.model_dir + "/{}-best.pt".format(experiment_fn))
                        
        if ((epoch+1)%50)==0 and args.save_model is not None:
            experiment_fn = args.save_model
            torch.save(model.state_dict(),
                       args.model_dir + "/{}-epoch{}.pt".format(experiment_fn,epoch))

    # Save final model
    if args.save_model is not None:
        experiment_fn = args.save_model
        torch.save(model.state_dict(),
                   args.model_dir + "/{}.pt".format(experiment_fn))


if __name__ == '__main__':
    main()



