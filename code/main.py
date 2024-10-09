# -*- coding: utf-8 -*

import random
import numpy as np
import torch
import torch.optim as optim
import argparse
import pickle
import logging
from utils import *
from model import *

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)

args = argparse.ArgumentParser(description='Argparse for compound-protein interactions prediction')
args.add_argument('-dataset', type=str, default='human', help='choose a dataset')
args.add_argument('-mode', type=str, default='gpu', help='gpu/cpu')
args.add_argument('-cuda', type=str, default='0', help='visible cuda devices')
args.add_argument('-verbose', type=int, default=1, help='0: do not output log in stdout, 1: output log')

# Hyper-parameter
args.add_argument('-lr', type=float, default=0.0005, help='init learning rate')
args.add_argument('-step_size', type=int, default=10, help='step size of lr_scheduler')
args.add_argument('-gamma', type=float, default=0.5, help='lr weight decay rate')
args.add_argument('-batch_size', type=int, default=16, help='batch size')
args.add_argument('-num_epochs', type=int, default=20, help='number of epochs')
args.add_argument('-dropout', type=float, default=0.1)
args.add_argument('-alpha', type=float, default=0.1, help='LeakyReLU alpha')

# bivae
args.add_argument('-k', type=int, default=20, help='dimension of latent factors')
args.add_argument('-encoder_structure', type=list, default=[40], help='the number of neurons per layer of encoders for BiVAE')
args.add_argument('-likelihood', type=str, default='pois', help='the likelihood function used for modeling the observations')
args.add_argument('-act_fn', type=str, default='relu', help='name of the activation function used between hidden layers of the auto-encoder')

# GIN layer
args.add_argument('-gin_layers', type=int, default=3, help='the number of linear layers in the neural network')
args.add_argument('-num_mlp_layers', type=int, default=3, help='the number of linear layers in mlps')
args.add_argument('-hidden_dim', type=int, default=50, help='dimension of hidden units at all layers')
args.add_argument('-neighbor_pooling_type', type=str, default='mean', help='how to aggregate neighbors (sum, mean, or max)')
args.add_argument('-graph_pooling_type', type=str, default='sum', help='how to aggregate entire nodes in a graph (sum, mean or max)')

args.add_argument('-comp_dim', type=int, default=80, help='dimension of compound atoms feature')
args.add_argument('-prot_dim', type=int, default=80, help='dimension of protein amino feature')
args.add_argument('-latent_dim', type=int, default=80, help='dimension of compound and protein feature')

args.add_argument('-window', type=int, default=5, help='window size of cnn model')
args.add_argument('-layer_cnn', type=int, default=3, help='number of layer in cnn model')


params, _ = args.parse_known_args()

def train(model, data_train, data_dev, data_test, compound_dict, protein_dict, device, params):
    criterion = nn.CrossEntropyLoss()
    best_res = 0
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    batch_size = params.batch_size
    for epoch in range(params.num_epochs):
        model.train()
        total_loss = 0
        pred_labels = []
        predictions = []
        labels = []
        optimizer.zero_grad()
        for no, data in enumerate(data_train):
            atoms, adj, fps, aminos, label = data[0], data[1], data[2], data[3].unsqueeze(0), data[4]
            c_id = compound_dict.get(no)
            p_id = protein_dict.get(no)
            pred = model(atoms, adj, aminos, fps, c_id, p_id, device)
            loss = criterion(pred.float(), label.view(label.shape[0]).long())
            loss = loss / batch_size
            ys = F.softmax(pred, 1).to('cpu').data.numpy()
            pred_labels += list(map(lambda x: np.argmax(x), ys))
            predictions += list(map(lambda x: x[1], ys))
            labels += label.cpu().numpy().reshape(-1).tolist()
            loss.backward()
            if no % batch_size==0 or no==len(data_train):
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()

        pred_labels = np.array(pred_labels)
        predictions = np.array(predictions)
        labels = np.array(labels)
        auc_train, f1_train, apur_train, precision_train, recall_train = classification_scores(labels, predictions, pred_labels)
        text = 'Train auc:{}, f1:{}, aupr:{}, precision:{}, recall:{}'.format(auc_train, f1_train, apur_train,precision_train, recall_train)
        print(text)
        save_result(save_path, filename, text)

        auc_dev, f1_dev, aupr_dev, precision_dev, recall_dev,_,_ = test(model, data_dev, compound_dict, protein_dict, len(data_train), device)
        text = 'Dev auc:{}, f1:{}, aupr:{}, precision:{}, recall:{}'.format(auc_dev, f1_dev, aupr_dev, precision_dev, recall_dev)
        print(text)
        save_result(save_path, filename, text)

        auc_test, f1_test, aupr_test, precision_test, recall_test, preds, labels = test(model, data_test, compound_dict, protein_dict, len(data_train)+len(data_dev), device)
        text = 'Test auc:{}, f1:{}, aupr:{}, precision:{}, recall:{}'.format(auc_test, f1_test, aupr_test, precision_test, recall_test)
        print(text)
        save_result(save_path, filename, text)

        if auc_dev > best_res:
            best_res = auc_dev
            res = [auc_test, f1_test, aupr_test, precision_test, recall_test, preds, labels]
            torch.save(model, 'model.pt')
        scheduler.step()
        if params.verbose:
            text = 'epoch{} loss:{}'.format(epoch + 1, total_loss)
            print(text)
            save_result(save_path, filename, text)
    return res


def test(model, test_data, compound_dict, protein_dict, length, device):
    model.eval()
    predictions = []
    pred_labels = []
    labels = []
    for no, data in enumerate(test_data):
        atoms, adj, fps, aminos, label = data[0], data[1], data[2], data[3].unsqueeze(0), data[4]
        c_id = compound_dict.get(no + length)
        p_id = protein_dict.get(no + length)
        with torch.no_grad():
            pred = model(atoms, adj, aminos, fps, c_id, p_id, device)
        ys = F.softmax(pred, 1).to('cpu').data.numpy()
        pred_labels += list(map(lambda x: np.argmax(x), ys))
        predictions += list(map(lambda x: x[1], ys))
        labels += label.cpu().numpy().reshape(-1).tolist()

    pred_labels = np.array(pred_labels)
    predictions = np.array(predictions)
    labels = np.array(labels)
    auc_value, f1_value, aupr_value, precison_value, recall_value = classification_scores(labels, predictions, pred_labels)
    return auc_value, f1_value, aupr_value, precison_value, recall_value, predictions, labels


if __name__ == '__main__':
    print(params)
    dataset = params.dataset
    data_dir = '../dataset/' + dataset + 'origin/'
    save_path = '../result/'+dataset + '/'
    filename = 'origin'
    torch.manual_seed(1234)
    random.seed(1234)
    if params.mode == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = params.cuda
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("cuda is not available!!!")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print('The code run on the', device)

    data = load_data(data_dir)
    data = data2tensor(data, device)
    train_data, data1 = split_dataset(data, 0.8)
    dev_data, test_data = split_dataset(data1, 0.5)
    print('train length', len(train_data))
    print('test length', len(test_data))
    print('validation length', len(dev_data))

    atom_dict = pickle.load(open(data_dir + 'atom_dict', 'rb'))
    amino_dict = pickle.load(open(data_dir + 'amino_dict', 'rb'))
    compound_dict = pickle.load(open(data_dir + 'compound_dict', 'rb'))
    protein_dict = pickle.load(open(data_dir + 'protein_dict', 'rb'))

    print('training bivae')
    data_matrix = np.load(data_dir+'data_matrix.npy')
    compound_nums = data_matrix.shape[0]
    protein_nums = data_matrix.shape[1]
    print('compound nums:', compound_nums)
    print('protein nums:', protein_nums)
    compound_encoder_structure = [protein_nums] + params.encoder_structure
    protein_encoder_structure = [compound_nums] + params.encoder_structure
    bivae = BiVAE(k=params.k,
                  compound_encoder_structure=compound_encoder_structure,
                  protein_encoder_structure=protein_encoder_structure,
                  act_fn=params.act_fn,
                  likelihood=params.likelihood)

    bivae = learn(bivae,data_matrix, epochs=100, batch_size=100, lr=0.001, beta_kl=1.0)
    print('bivae finish training!')
    model = BiBAECPI(bivae, len(atom_dict), len(amino_dict), params)
    model = model.to(device)
    res = train(model, train_data, dev_data, test_data, compound_dict, protein_dict, device, params)
    text = 'Finally test result of auc:{}, f1:{}, aupr:{}, precision:{}, recall:{}'.format(res[0], res[1], res[2], res[3], res[4])
    print(text)
    save_result(save_path, filename, text)



