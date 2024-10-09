# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import numpy as np
import itertools as it
import math
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

EPS = 1e-10
ACT = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
}
class BiVAE(nn.Module):
    def __init__(self, k, compound_encoder_structure, protein_encoder_structure, likelihood, act_fn):
        super(BiVAE, self).__init__()
        self.mu_theta = torch.zeros((protein_encoder_structure[0], k))  # n_compound * k
        self.mu_beta = torch.zeros((compound_encoder_structure[0], k))  # n_protein * k
        self.theta = torch.randn(protein_encoder_structure[0], k) * 0.01
        self.beta = torch.randn(compound_encoder_structure[0], k) * 0.01
        torch.nn.init.kaiming_uniform_(self.theta, a=np.sqrt(5))

        self.likelihood = likelihood
        self.act_fn = ACT.get(act_fn, None)

        # compound encoder
        self.compound_encoder = nn.Sequential()
        for i in range(len(compound_encoder_structure) - 1):
            self.compound_encoder.add_module(
                "fc{}".format(i),
                nn.Linear(compound_encoder_structure[i], compound_encoder_structure[i + 1]),
            )
            self.compound_encoder.add_module("act{}".format(i), self.act_fn)
        self.compound_mu = nn.Linear(compound_encoder_structure[-1], k)  # mu
        self.compound_std = nn.Linear(compound_encoder_structure[-1], k)

        # protein Encoder
        self.protein_encoder = nn.Sequential()
        for i in range(len(protein_encoder_structure) - 1):
            self.protein_encoder.add_module(
                "fc{}".format(i),
                nn.Linear(protein_encoder_structure[i], protein_encoder_structure[i + 1]),
            )
            self.protein_encoder.add_module("act{}".format(i), self.act_fn)
        self.protein_mu = nn.Linear(protein_encoder_structure[-1], k)  # mu
        self.protein_std = nn.Linear(protein_encoder_structure[-1], k)

    def encode_compound(self, x):
        h = self.compound_encoder(x)
        return self.compound_mu(h), torch.sigmoid(self.compound_std(h))

    def encode_protein(self, x):
        h = self.protein_encoder(x)
        return self.protein_mu(h), torch.sigmoid(self.protein_std(h))

    def decode_compound(self, theta, beta):
        h = theta.mm(beta.t())
        return torch.sigmoid(h)

    def decode_protein(self, theta, beta):
        h = beta.mm(theta.t())
        return torch.sigmoid(h)

    def reparameterize(self, mu, std):
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, x, compound=True, beta=None, theta=None):
        if compound:
            mu, std = self.encode_compound(x)
            theta = self.reparameterize(mu, std)
            return theta, self.decode_compound(theta, beta), mu, std
        else:
            mu, std = self.encode_protein(x)
            beta = self.reparameterize(mu, std)
            return beta, self.decode_protein(theta, beta), mu, std

    def loss(self, x, x_, mu, std, kl_beta):
        # Likelihood
        ll_choices = {
            "bern": x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS),
            "gaus": -(x - x_) ** 2,
            "pois": x * torch.log(x_ + EPS) - x_,
        }

        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError("Supported likelihoods: {}".format(ll_choices.keys()))
        ll = torch.sum(ll, dim=1)

        # KL term
        kld = -0.5 * (1 + 2.0 * torch.log(std) - mu.pow(2) - std.pow(2))
        kld = torch.sum(kld, dim=1)

        return torch.mean(kl_beta * kld - ll)


def learn(bivae, data_matrix, epochs, batch_size, lr, beta_kl, device=torch.device("cpu"),dtype=torch.float32):
    compound_params = it.chain(bivae.compound_encoder.parameters(),
                               bivae.compound_mu.parameters(),
                               bivae.compound_std.parameters())
    protein_params = it.chain(bivae.protein_encoder.parameters(),
                              bivae.protein_mu.parameters(),
                              bivae.protein_std.parameters())
    c_optimizer = torch.optim.Adam(params=compound_params, lr=lr)
    p_optimizer = torch.optim.Adam(params=protein_params, lr=lr)
    x = data_matrix
    tx = x.T
    c_idx = np.arange(x.shape[0])
    p_idx = np.arange(tx.shape[0])
    best_bivae = None
    best_loss = math.inf
    for epoch in range(epochs):
        # protein side
        p_sum_loss = 0
        for i in range(math.ceil(tx.shape[0] / batch_size)):
            p_ids = p_idx[i * batch_size:(i + 1) * batch_size]
            p_batch = tx[p_ids, : ]
            p_batch = torch.tensor(p_batch, dtype=dtype, device=device)
            beta, p_batch_, p_mu, p_std = bivae(p_batch, compound=False, theta=bivae.theta)
            p_loss = bivae.loss(p_batch, p_batch_, p_mu, p_std, beta_kl)
            p_optimizer.zero_grad()
            p_loss.backward()
            p_optimizer.step()

            p_sum_loss += p_loss.item()
            beta, _, p_mu, _ =  bivae(p_batch, compound=False, theta=bivae.theta)
            bivae.beta.data[p_ids] = beta.data
            bivae.mu_beta.data[p_ids] = p_mu.data

        # compound side
        c_sum_loss = 0
        for i in range(math.ceil(x.shape[0] / batch_size)):
            c_ids = c_idx[i * batch_size:(i + 1) * batch_size]
            c_batch = x[c_ids, :]
            c_batch = torch.tensor(c_batch, dtype=dtype, device=device)
            theta, c_batch_, c_mu, c_std = bivae(c_batch, compound=True, beta=bivae.beta)
            c_loss = bivae.loss(c_batch, c_batch_, c_mu, c_std, beta_kl)
            c_optimizer.zero_grad()
            c_loss.backward()
            c_optimizer.step()

            c_sum_loss += c_loss.item()
            theta, _, c_mu, _ = bivae(c_batch, compound=True, beta=bivae.beta)
            bivae.theta.data[c_ids] = theta.data
            bivae.mu_theta.data[c_ids] = c_mu.data

        if p_sum_loss+c_sum_loss < best_loss :
            best_loss = p_sum_loss+c_sum_loss
            best_bivae = bivae



    # infer mu_beta
    for i in range(math.ceil(tx.shape[0] / batch_size)):
        p_ids = p_idx[i * batch_size:(i + 1) * batch_size]
        p_batch = tx[p_ids, :]
        p_batch = torch.tensor(p_batch, dtype=dtype, device=device)
        beta, _, p_mu, _ = best_bivae(p_batch, compound=False, theta=bivae.theta)
        best_bivae.mu_beta.data[p_ids] = p_mu.data

    # infer mu_theta
    for i in range(math.ceil(x.shape[0] / batch_size)):
        c_ids = c_idx[i * batch_size:(i + 1) * batch_size]
        c_batch = x[c_ids, :]
        c_batch = torch.tensor(c_batch, dtype=dtype, device=device)
        theta, _, c_mu, _ = best_bivae(c_batch, compound=True, beta=bivae.beta)
        best_bivae.mu_theta.data[c_ids] = c_mu.data

    return best_bivae


class ApplyNodeFunc(nn.Module):
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.InstanceNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.InstanceNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):

        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.InstanceNorm1d(hidden_dim))

        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)


        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer


class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim,self.hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved

        conved = conved.permute(0,2,1)
        return conved


class BiBAECPI(nn.Module):
    def __init__(self, bivae, n_atom, n_amino, params):
        super(BiBAECPI, self).__init__()
        comp_dim, prot_dim, gin_layers, num_mlp_layers, dropout, alpha, window, layer_cnn, latent_dim, hidden_dim, k,\
            neighbor_pooling_type, graph_pooling_type = params.comp_dim, params.prot_dim, params.gin_layers, \
            params.num_mlp_layers, params.dropout, params.alpha, params.window, params.layer_cnn, params.latent_dim, \
            params.hidden_dim, params.k, params.neighbor_pooling_type, params.graph_pooling_type

        self.embedding_layer_atom = nn.Embedding(n_atom + 1, comp_dim)  # nn.Embedding(n,m)
        self.embedding_layer_amino = nn.Embedding(n_amino + 1, prot_dim)

        self.bivae = bivae
        self.dropout = dropout
        self.alpha = alpha
        self.layer_cnn = layer_cnn

        self.gin = GIN(num_layers=gin_layers, num_mlp_layers=num_mlp_layers, input_dim=comp_dim, hidden_dim=hidden_dim, output_dim=latent_dim, final_dropout=dropout,
          learn_eps=False, graph_pooling_type=graph_pooling_type, neighbor_pooling_type=neighbor_pooling_type)

        self.encoder = Encoder(prot_dim, hid_dim=latent_dim, n_layers=layer_cnn, kernel_size=2 * window + 1, dropout=dropout, device=torch.device('cuda'))

        self.fp0 = nn.Parameter(torch.empty(size=(1024, latent_dim)))
        nn.init.xavier_uniform_(self.fp0, gain=1.414)
        self.fp1 = nn.Parameter(torch.empty(size=(latent_dim, k)))
        nn.init.xavier_uniform_(self.fp1, gain=1.414)

        self.trans_comp = nn.Linear(latent_dim, k)
        self.trans_pro = nn.Linear(latent_dim, k)
        self.in_norm = nn.InstanceNorm1d(num_features=1)

        self.out = nn.Linear(k*3, 2)
        self.drop_out = nn.Dropout(p=self.dropout)

    def comp_gin(self, atoms, adj, device):
        # GIN
        atoms_vector = self.embedding_layer_atom(atoms)
        adj = np.array(adj.cpu())
        a = np.nonzero(adj)
        g = dgl.graph(a).to(device)
        atoms_vector = self.gin(g, atoms_vector)

        return atoms_vector

    def prot_encoder(self, amino):
        amino_vector = self.embedding_layer_amino(amino)
        amino_vector = self.encoder(amino_vector)
        amino_vector = F.leaky_relu(amino_vector, self.alpha)
        return amino_vector  # (batch_size, lenth, dim)

    def forward(self, atoms, adjacency, amino, fps, c_id, p_id, device):
        atoms_vector = self.comp_gin(atoms, adjacency, device)

        amino_vector = self.prot_encoder(amino)
        atoms_vector = self.in_norm(atoms_vector).squeeze(0)
        amino_vector = self.in_norm(amino_vector)
        amino_vector = torch.sum(amino_vector, dim=1).squeeze(0)

        atoms_vector = self.trans_comp(atoms_vector)
        amino_vector = self.trans_pro(amino_vector)

        theta_c = self.bivae.mu_theta[c_id].to(device)
        beta_p = self.bivae.mu_beta[p_id].to(device)
        fea_com = theta_c * atoms_vector
        fea_pro = beta_p * amino_vector
        fea_com = F.leaky_relu(fea_com, 0.1)
        fea_pro = F.leaky_relu(fea_pro, 0.1)

        fps_vector = F.leaky_relu(torch.matmul(fps, self.fp0), 0.1)
        fps_vector = F.leaky_relu(torch.matmul(fps_vector, self.fp1), 0.1)

        fusion_feature = torch.cat((fea_com, fea_pro, fps_vector))
        result = self.out(fusion_feature)
        result = result.reshape(1,2)
        return result

