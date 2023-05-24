import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import torch.nn.functional as F

from circuit_explorer.utils import load_config
from circuit_explorer.receptive_fields import receptive_field, receptive_field_for_unit
from circuit_explorer.target import layer_saver
from dataset.Dataset_Utils import ImageNetKaggle, ImageNetReceptiveField, ImageNetClusters
from models.Masked_AlexNet import AlexNetMasking

import numpy as np
from tqdm import tqdm
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns

import os
from PIL import Image
import json

from HDBScan_Clustering import cluster

import random
import warnings

warnings.filterwarnings('ignore')

def adjust_learning_rate(optimizer, epoch, lr):
    for (drop, step) in zip(lr_drops, lr_schedule):
        if (epoch >= step): lr = lr * drop
        else: break
    for param_group in optimizer.param_groups: param_group['lr'] = lr

def compute_remaining_weights(masks):
    return 1 - sum(float((m == 0.).sum()) for m in masks) / sum(m.numel() for m in masks)

def compute_loss(model, output, target, lmbda):

    class_loss = F.binary_cross_entropy_with_logits(output, target) 

    masks = [m.mask for m in model.mask_modules]
    all_masks = [[m.weight_mask, m.bias_mask] for m in model.mask_modules]
    all_masks = reduce(lambda a,b: a+b, weight_masks)
    entries_sum = sum(m.sum() for m in masks)
    l0_loss = lmbda * entries_sum 

    loss = class_loss + l0_loss
    return loss, (class_loss, l0_loss)

def compute_hard_mask_accuracy(model, dataloader, target):
    model.ticket = True # sets model to use hard mask
    model.eval()

    acc_batch = []
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(device)
            target = target.to(device).float()
            output = model.run_to_conv_layer(data, layer)
            output = output[:, unit, position[0], position[1]]
            
            ## COMPUTE BATCH AND RUNNING ACCURACY WITH HARD MASKS
            pred = (output > 0).float()
            batch_correct = pred.eq(target.data.view_as(pred)).sum().to("cpu")
            curr_acc = batch_correct/len(target)
            acc_batch.append(curr_acc.item())

    model.ticket = False ## sets model to use soft mask
    model.train()
    return np.mean(acc_batch)

        
def train(FLAGS, model, dataloader, optimizer, temp_increase, layer='10', unit=255, position=[6, 6]):

    remaining_weights = 1
    hard_acc = 0
    device=FLAGS.device

    ## OUTER TRAINING LOOP
    for epoch in range(FLAGS.epochs):
        # if epoch % 80 == 0:
        #   torch.save(model.state_dict(), f'outputs/model_epoch={epoch}.pt')

        model.train()

        if epoch > 0: model.set_temp(model.temp * temp_increase) ## changes sigmoid temp
        adjust_learning_rate(optimizer, epoch, FLAGS.lr)

        acc_batch = []
        
        ## INSIDE EPOCH LOOP
        pbar = tqdm(dataloader)
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(device)
            target = target.to(device).float()
            optimizer.zero_grad()

            output = model.run_to_conv_layer(data, layer)
            print(output.shape)
            output = output[:, unit, position[0], position[1]]

            loss, (class_loss, l0_loss) = compute_loss(model, output, target, FLAGS.lmbda)
            loss.backward()
            optimizer.step()

            ## COMPUTE BATCH AND RUNNING ACCURACY WITH SOFT MASKS
            pred = (output > 0).float()
            batch_correct = pred.eq(target.data.view_as(pred)).sum().to("cpu")
            curr_acc = batch_correct/len(target)
            acc_batch.append(curr_acc.item())
            acc = np.mean(acc_batch)

            ## UPDATE TQDM BAR
            pbar.set_postfix(temp=model.temp, remaining_weights=remaining_weights, acc=acc, hard_acc=hard_acc,
                            class_loss=class_loss.item(), l0_loss=l0_loss.item(), epoch=epoch)
        
        remaining_weights = compute_remaining_weights(masks) ## not the best metric tbh, not sure why include
        hard_acc = compute_hard_mask_accuracy(model, dataloader, target) ## accuracy with hard masks on all training data

def main(FLAGS):

    ## LOAD DATASET AND ORIGINAL MODEL
    device = FLAGS.device
    layer = FLAGS.layer
    unit = FLAGS.unit

    position = [6, 6] ## hard coded because resize to effective receptor field

    config_file = FLAGS.config_file
    config = load_config(config_file)

    unmasked_model = config.model
    unmasked_model = unmasked_model.to(device)
    unmasked_model.eval().cuda()

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    preprocessing = transforms.Compose(
              [
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize(mean, std),
              ]
          )

    all_recep_field_params = receptive_field(unmasked_model.features, (3,224,224))
    recep_field = receptive_field_for_unit(all_recep_field_params, layer, position)

    imagenet_train_dataset = ImageNetReceptiveField(FLAGS.data_path, "train", preprocessing, 
                                                    recep_field=recep_field)
    
    ## CLUSTERING
    layer_num = int(layer.split('.')[-1])
    target_layer_activations = torch.load('outputs/layer_{layer_num}/layer_{layer_num}_image_net_1000.pt') 
    values, idxs, labels = cluster(target_layer_activations, layer, unit, model, 
                                    imagenet_train_dataset, min_cluster_size=FLAGS.min_cluster_size)
    # values = sorted activations; idxs = sorted indices (by activation); labels = cluster labels

    ## MAKE SUBSET DATASET [Assuming 2 clusters + noise]
    both_cluster_indices = idxs[np.concatenate((np.where(labels == 0)[0], np.where(labels == 1)[0]))]
    both_cluser_labels = [0] * len(np.where(labels == 0)[0]) + [1] * len(np.where(labels == 1)[0])
    
    if FLAGS.switch_cluster_labels:
        both_cluser_labels = 1 - np.array(both_cluser_labels) ## if you want to switch labels

    
    clusters_indices = {clust_ind:clust_lab for clust_ind, clust_lab in zip(both_cluster_indices, both_cluser_labels)}
    torch.save()
    imagenet_cluster_dataset = ImageNetClusters(FLAGS.data_path, "train", clusters=clusters_indices, 
                                                transform=preprocessing, recep_field=recep_field)
    trainset_both_clusters = torch.utils.data.Subset(imagenet_cluster_dataset, both_cluster_indices)
    dataloader_both_clusters = DataLoader(
            trainset_both_clusters,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True
        )

    ## MAKE MASKED MODEL
    masked_model = AlexNetMasking(masked_layers=FLAGS.mask_layers)
    model_statedict = torch.load(FLAGS.model_state_dict)
    masked_model.load_state_dict(model_statedict)
    masked_model.model_init_mask()

    masked_model = masked_model.to(device)

    iters_per_reset = FLAGS.epochs-1
    temp_increase = FLAGS.final_temp**(1./iters_per_reset)

    trainable_params = filter(lambda p: p.requires_grad, masked_model.parameters())
    num_params = sum([p.numel() for p in trainable_params])

    print(f"Total number of parameters: {num_params}")

    mask_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask' in p[0], model.named_parameters()))
    mask_optim = optim.SGD(mask_params, lr=lr)

    ## TRAIN MASKED MODEL
    masked_model.ticket = False
    masked_model.temp = 1

    train(FLAGS, masked_model, dataloader_both_clusters, mask_optim, temp_increase=temp_increase)

    ## SAVE MASKED MODEL
    reverse = "_rev" if FLAGS.switch_cluster_labels else ""
    torch.save(model.state_dict(), f'outputs/layer_{layer_num}/model_unit_{unit}_mask{reverse}.py')

    ## PRINT PROPORTION OF NON-ZERO MASKS
    masks = [(m.weight_mask > 0).float() for m in model.mask_modules]
    print("Proportion of non-zero masks", 1 - sum(float((m == 0.).sum()) for m in masks) / sum(m.numel() for m in masks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='Subnetwork_Training',
                description='trains masks for alexnet specific task')
    parser.add_argument("--layer", default="features.10", type=str, help="specified layer to collect top k activations at")
    parser.add_argument('--unit', default=255, type=int, help='what unit to train subnetwork for')
    parser.add_argument("--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("--epochs", default=160, type=int, help="number epochs")
    parser.add_argument("--device", default="cuda:0", type=str, help="device to run on")
    parser.add_argument("--lr", default=0.1, type=int, help="the learning rate of bce loss")
    parser.add_argument("--seed", default=1234, type=int, help="the seed for randomization")
    parser.add_argument("--lr-schedule", nargs='+', default=[130, 150], help="learning rate schedule")
    parser.add_argument("--lr-drops", nargs='+', default=[0.1, 0.1], help="learning rate gamma drops")
    parser.add_argument("--decay", default=0.0001, type=float, help="weight decay for reg")
    parser.add_argument("--lmdba", default=1e-6, type=float, help="lambda scaling for l0 regularization")
    parser.add_argument("--final_temp", default=200, type=int, help="final temperature of the sigmoid function")
    parser.add_argument("--mask-initial-value", default=0, type=int, help="mask value initial values (pre-sigmoid)")
    parser.add_argument("--min-cluster-size", default=50, type=int, help="HDB clustering min cluster size")
    parser.add_argument("--device", default="cuda:0", type=str, help="device to run training on")
    parser.add_argument("--switch-cluster-labels", default=False, type=bool, 
        help="switches the cluster labels to find subnetwork of other class")
    parser.add_argument("--config-file", default="configs/alexnet_sparse_config.py", type=str,
        help="config file specifying model to run (default: alexnet-sparse)"
        )
    parser.add_argument("--data-path", default="image_data/imagenet", type=str, 
        help="where the imagenet data is located")
    parser.add_argument("--model-state-dict", default="./models/alexnet_sparse_statedict.pt", type=str, 
        help="model state dictionary with original weights to use to initialize masked model"
        )
    parser.add_argument("--mask-layers", default=[0, 3, 6, 8, 10], nargs="+", 
        help="list of layers that should be able to be masked in the subnetwork"
        )
    ## RIGHT NOW ONLY SUPPORTS ALEX-NET/ALEX-NET-SPARSE

    FLAGS = parser.parse_args()
    assert(len(FLAGS.lr_schedule) == len(FLAGS.lr_drops)) ## gammas and schedule should be same size

    main(FLAGS)