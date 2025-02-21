# -*- coding: utf-8 -*
import os
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score


def load_data(dir_input):
    compounds = np.load(dir_input + 'compounds.npy', allow_pickle=True)
    adjacencies = np.load(dir_input + 'adjacencies.npy', allow_pickle=True)
    fingerprint = np.load(dir_input + 'fingerprint.npy', allow_pickle=True)
    proteins = np.load(dir_input + 'proteins.npy', allow_pickle=True)
    interactions = np.load(dir_input + 'interactions.npy', allow_pickle=True)
    data_pack = [compounds, adjacencies, fingerprint, proteins, interactions]
    return data_pack


def fps2number(arr):
    new_arr = np.zeros((arr.shape[0], 1024))
    for i, a in enumerate(arr):
        new_arr[i, :] = np.array(list(a), dtype=int)
    return new_arr

def data2tensor(data, device):
    atoms = [torch.LongTensor(d).to(device) for d in data[0]]
    adjacencies = [torch.FloatTensor(d).to(device) for d in data[1]]
    fps = fps2number(data[2])
    fingerprint = [torch.FloatTensor(d).to(device) for d in fps]
    amino = [torch.LongTensor(d).to(device) for d in data[3]]
    interactions = [torch.LongTensor(d).to(device) for d in data[4]]
    dataset = list(zip(atoms, adjacencies, fingerprint, amino, interactions))
    return dataset


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def classification_scores(label, pred_score, pred_label):
    label = label.reshape(-1)
    pred_score = pred_score.reshape(-1)
    pred_label = pred_label.reshape(-1)
    auc = roc_auc_score(label, pred_score)
    precision, recall, _ = precision_recall_curve(label, pred_score)
    aupr = metrics.auc(recall, precision)
    f1 = f1_score(label, pred_label)

    precision = precision_score(label, pred_label)
    recall = recall_score(label, pred_label)

    return np.round(auc, 6), np.round(f1, 6), np.round(aupr, 6), np.round(precision, 6), np.round(recall, 6)


def get_ids(compound_dict, protein_dict, ids):
    c_ids, p_ids = [], []
    for i in range(len(ids)):
        c_id = compound_dict.get(ids[i])
        p_id = protein_dict.get(ids[i])
        c_ids.append(c_id)
        p_ids.append(p_id)
    return c_ids, p_ids


def save_result(path, filename, text):
    if not os.path.exists(path):     
        os.makedirs(path)
    name = path + filename + '.txt'    #create txt file
    file = open(name,'a')
    file.write(text+'\n')        #write result information
    file.close()


