#!/usr/bin/env python
# coding: utf-8

import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import os
from PIL import Image
import json

from circuit_explorer.utils import load_config
from circuit_explorer.utils import get_layers_from_model

from torchvision import transforms, datasets
from circuit_explorer.data_loading import rank_image_data, single_image_data
from circuit_explorer.receptive_fields import receptive_field, receptive_field_for_unit

from collections import defaultdict
from circuit_explorer.target import layer_saver
from heapq import heappush, heappushpop

from circuit_explorer.target import layer_saver
from circuit_explorer.data_loading import default_unnormalize

import hdbscan
from dataset.Dataset_Utils import ImageNetReceptiveField, ImageNet2

position = [6, 6]

def k_layer_activations_from_dataloader(layers,dataloader,model,num_units=[384],k=300,position=[6,6],batch_size=64, device="cuda:0"):
    
    device = next(model.parameters()).device
    topk_activations = {}
    layer_num_units = {}
    
    layer_activations = {}
    if isinstance(layers,str):
        layers = [layers]
    for i in layers:
        num_units = get_layers_from_model(model)[i].out_channels
        layer_num_units[i] = num_units
        for unit in range(num_units):
            topk_activations[f'{i}_{unit}'] = []

    for dl_idx, (data, labels) in enumerate(tqdm(dataloader)):  
        ds_idxs = list(range(dl_idx*batch_size, (dl_idx+1)*batch_size))
        
        images = data.to(device)
        with layer_saver(model, layers) as extractor:
            batch_layer_activations = extractor(images)
            for i in layers:
                for unit in range(layer_num_units[i]):
                    unit_activations = list(batch_layer_activations[i][:, unit, position[0], position[1]].detach().to('cpu').numpy())
                    tuples_to_add = zip(unit_activations, ds_idxs)
                    
                    for tup in tuples_to_add:
                        if len(topk_activations[f'{i}_{unit}']) < k:
                            heappush(topk_activations[f'{i}_{unit}'], tup)
                        else:
                            heappushpop(topk_activations[f'{i}_{unit}'], tup)

    return topk_activations


def cluster(target_layer_activations, layer, unit, model, imagenet_train_dataset, min_cluster_size=50, device="cuda:0"):
    original_activations = target_layer_activations[f'{layer}_{unit}']
    activations, indices = list(zip(*original_activations))

    sorted_idxs = np.flip(np.argsort(activations))
    sorted_activations = np.array(activations)[sorted_idxs]
    sorted_indices = np.array(indices)[sorted_idxs]

    values, idxs = sorted_activations, sorted_indices
    
    layer_activations = {}
    if isinstance(layer, str):
        layers = [layer]
    for i in layers:
        layer_activations[i] = []

    for i in range(len(idxs)):
        data = imagenet_train_dataset[idxs[i]]
        images = data[0].to(device)

        with layer_saver(model, layers) as extractor:
            batch_layer_activations = extractor(images.unsqueeze(0))
            for i in layers:
                layer_activations[i].append(batch_layer_activations[i].detach().to('cpu'))

    for i in layers:     
        layer_activations[i] = torch.cat(layer_activations[i])

    cluster_points = layer_activations[layer][:,:,position[0],position[1]]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(cluster_points.numpy())
    return labels, values, idxs

def get_unit_clustering_statistics(target_layer_activations, layer, model, imagenet_train_dataset, num_filters=256, min_cluster_size=50, device="cuda:0"):
    unit_dict = {}
    for u in tqdm(range(num_filters)):
        labels, _, _ = cluster(target_layer_activations, layer, u, model, imagenet_train_dataset, min_cluster_size=min_cluster_size, device=device)

#         num_0s = len(np.where(labels==0)[0])
#         num_1s = len(np.where(labels==1)[0])
#         num_neg_1s = len(np.where(labels==-1)[0])
        unit_dict[u] = {val:len(np.where(labels==val)[0]) for val in np.unique(labels)}
    return unit_dict

def cluster_and_plot_unit(target_layer_activations, layer, model, imagenet_train_dataset, unit, name, min_cluster_size=50, device="cuda:0"):
    labels, values, idx = cluster(target_layer_activations, layer, unit, model, imagenet_train_dataset, min_cluster_size=min_cluster_size, device=device)

    plt.figure(figsize=(12, 12))
    num_images = 16
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        if i < int(num_images/2):
            plt.imshow(default_unnormalize(imagenet_train_dataset[random.choice(idxs[np.where(labels == 0)])][0]).numpy().transpose((1, 2, 0)))
        else:
            plt.imshow(default_unnormalize(imagenet_train_dataset[random.choice(idxs[np.where(labels == 1)])][0]).numpy().transpose((1, 2, 0)))
        plt.axis('off')
        plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        plt.tight_layout()
    
    plt.savefig(name)
            

def main(FLAGS):
    
    device = FLAGS.device

    config_file = FLAGS.config_file
    config = load_config(config_file)

    model = config.model
    model = model.to(device)

    all_layers = get_layers_from_model(model)
    all_layers.keys()

    model.eval().cuda()
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

    position = [6,6]
    # layer = 'features.8'
    # unit = 53

    #dog/metal feature
    layer = FLAGS.layer

    all_recep_field_params = receptive_field(model.features, (3,224,224))
    recep_field = receptive_field_for_unit(all_recep_field_params, layer, position)

    # imagenet2_train_dataset = ImageNet2('image_data/imagenet_2', preprocessing, 
    #                                                 recep_field=recep_field)

    # imagenet2_train_dataloader = DataLoader(
    #             imagenet2_train_dataset,
    #             batch_size=128,
    #             num_workers=1,
    #             shuffle=False,
    #             drop_last=False,
    #             pin_memory=True
    #         )

    imagenet_train_dataset = ImageNetReceptiveField(FLAGS.data_path, "train", preprocessing, 
                                                    recep_field=recep_field)

    imagenet_train_dataloader = DataLoader(
                imagenet_train_dataset,
                batch_size=FLAGS.batch_size,
                num_workers=8,
                shuffle=False,
                drop_last=False,
                pin_memory=True
            )

    target_layer_activations = k_layer_activations_from_dataloader(layer,imagenet_train_dataloader,
                                                                model, k=FLAGS.k, batch_size=FLAGS.batch_size, 
                                                                device=device)

    num_filters = all_layers[layer].out_channels
    unit_dict = get_unit_clustering_statistics(target_layer_activations, layer, model, 
                                            imagenet_train_dataset, num_filters=num_filters, min_cluster_size=FLAGS.min_cluster_size, 
                                            device=device)

    layer_num = int(layer.split('.')[-1])
    torch.save(target_layer_activations, f'outputs/layer_{layer_num}/layer_{layer_num}_image_net_{FLAGS.k}.pt')
    torch.save(unit_dict, f'outputs/layer_{layer_name}/cluster_unit_stats_k={FLAGS.k}_mcs={FLAGS.min_cluster_size}.pt')

    if FLAGS.plot:
        for unit in units:
            cluster_and_plot_unit(target_layer_activations, layer, model, imagenet_train_dataset, unit, 
                                    f'outputs/layer_{layer_name}/unit_{unit}_clusters.png', min_cluster_size=FLAGS.min_cluster_size, 
                                    device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='HDBScan_Clustering',
                description='clusters specified layer and gives unit features')
    parser.add_argument("--layer", default="features.10", type=str, help="specified layer to collect top k activations at")
    parser.add_argument("--k", default=1000, type=int, help="number of activations to collect and cluster with")
    parser.add_argument("--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("--min-cluster-size", default=50, type=int, help="HDB clustering min cluster size")
    parser.add_argument("--device", default="cuda:0", type=str, help="device to run on")
    parser.add_argument("--plot", default=False, type=bool, help="whether to plot and save given units")
    parser.add_argument('--units', nargs='+', help='which units to plot the two classes of')
    parser.add_argument("--config-file", default="configs/alexnet_sparse_config.py", type=str,
        help="config file specifying model to run (default: alexnet-sparse)"
        )
    parser.add_argument("--data-path", default="image_data/imagenet", type=str, 
        help="where the imagenet data is located")

    FLAGS = parser.parse_args()
    main(FLAGS)