from sklearn.metrics import f1_score, classification_report
from numpy import vstack
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F

# from core.utils import to_device


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class ContEmbDataset(Dataset):
    def __init__(self, X, Y, cat_col, cont_col):
        X = X.copy()
        # categorical columns
        self.cat_X = X.loc[:, cat_col].copy().values.astype(np.int64)
        # numerical columns
        self.con_X = X.loc[:, cont_col].copy().copy().values.astype(np.float32)
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.cat_X[idx], self.con_X[idx], self.y[idx]


class ContEmbModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont, target_dim):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        self.n_emb = sum(e.embedding_dim for e in self.emb_layers)  # length of all embeddings combined
        self.n_cont = n_cont
        self.target_dim = target_dim
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 800)
        self.lin2 = nn.Linear(800, 280)
        self.lin3 = nn.Linear(280, self.target_dim)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(800)
        self.bn3 = nn.BatchNorm1d(280)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def forward(self, x_cat, x_cont):
        x1 = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
        x1 = torch.cat(x1, 1)
        x1 = self.emb_drop(x1)
        x2 = self.bn1(x_cont) #1st BN layer (normalized cont data)
        x = torch.cat([x1, x2], 1) #concat 2 inputs: [embeds, cont_data]
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x

    # def forward(self, cont_data, cat_data):
    #
    #     if self.no_of_embs != 0:
    #         x = [emb_layer(cat_data[:, i])
    #              for i, emb_layer in enumerate(self.emb_layers)]
    #         x = torch.cat(x, 1)
    #         x = self.emb_dropout_layer(x)
    #
    #     if self.no_of_cont != 0:
    #         normalized_cont_data = self.first_bn_layer(cont_data)
    #
    #         if self.no_of_embs != 0:
    #             x = torch.cat([x, normalized_cont_data], 1)
    #         else:
    #             x = normalized_cont_data
    #
    #     for lin_layer, dropout_layer, bn_layer in \
    #             zip(self.lin_layers, self.droput_layers, self.bn_layers):
    #         x = F.relu(lin_layer(x))
    #         x = bn_layer(x)
    #         x = dropout_layer(x)
    #
    #     x = self.output_layer(x)
    #
    #     return x

    def get_embeddings(self, cat_features: list,
                       emb_dims: list,
                       label_encoders: dict) -> dict:

        """
        :param cat_features - list, cat features names
        :param emb_dims - list, size for each emb layer
        :param label_encoders - dict, LabelEncoder for each cat feature
        :return - dict, emb for each cat feature
        """

        embeddings_dict = {}

        for i in range(len(cat_features)):
            feature = cat_features[i]
            encoder = label_encoders[feature]
            emb_layer = self.emb_layers[i]
            emb_dim = emb_dims[i][0]
            embeddings_dict[feature] = {
                k: v.detach().numpy() for k, v in zip(encoder.classes_,
                                                      emb_layer(torch.tensor(
                                                          [j for j in
                                                           range(emb_dim)])))}

        return embeddings_dict

class Calculate():
    def __init__(self, data, target_col, cat_col, cont_col, no_of_epochs=5, batch_size=256):
        self.data = data
        self.target_col = target_col
        self.cat_col = cat_col
        self.cont_col = cont_col
        #TODO put as params
        self.no_of_epochs = no_of_epochs
        self.batch_size = batch_size

    def preprocess(self) -> object:
        data = self.data[self.cont_col + self.cat_col + self.target_col]
        data.fillna(0, inplace=True)

        label_encoders = {}
        for col in self.cat_col:
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col])
            data[col] = data[col].astype('category')

        X = data[self.cont_col + self.cat_col]
        Y = LabelEncoder().fit_transform(data[self.target_col])
        return X, Y

    @staticmethod
    def split_data(X, Y):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    def get_emb_sizes(self, X):
        emb_cols_dict = {n: len(col.cat.categories) for n, col in X[self.cat_col].items() if len(col.cat.categories) > 2}
        emb_sizes = [(c, min(50, (c + 1) // 2)) for _, c in emb_cols_dict.items()]
        return emb_sizes

    @staticmethod
    def get_optimizer(model, lr=0.001, wd=0.0):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)
        return optim

    @staticmethod
    def train_model(model, optim, train_dl):
        model.train()
        total = 0
        sum_loss = 0
        for x1, x2, y in train_dl:
            batch = y.shape[0]
            output = model(x1, x2)
            loss = F.cross_entropy(output, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += batch
            sum_loss += batch * (loss.item())

            print('weights')
            print(torch.sum(model.lin3.weight.data))
            trained_weights = model.lin3.weight.data
        #     print(trained_weights)
        return sum_loss / total, trained_weights

    @staticmethod
    def val_loss(model, valid_dl):
        model.eval()
        total = 0
        sum_loss = 0
        correct = 0
        y_true, y_pred = list(), list()
        for x1, x2, y in valid_dl:
            current_batch_size = y.shape[0]
            out = model(x1, x2)
            loss = F.cross_entropy(out, y)
            sum_loss += current_batch_size * (loss.item())
            total += current_batch_size
            pred = torch.max(out, 1)[1]
            correct += (pred == y).float().sum().item()

            y = y.reshape((len(y), 1))
            pred = pred.reshape((len(pred), 1))
            # store
            y_pred.append(pred)
            y_true.append(y)
        print(len(y_pred), len(y_true))
        print("valid loss %.3f and accuracy %.3f" % (sum_loss / total, correct / total))
        y_true, y_pred = vstack(y_true), vstack(y_pred)
        return y_true, y_pred

    def train_loop(self, train_dl, test_dl, model, epochs, lr=0.01, wd=0.0):
        optim = self.get_optimizer(model, lr=lr, wd=wd)
        for epoch in range(epochs):
            loss, weights = self.train_model(model, optim, train_dl)
            print("training loss: ", loss)
            y_true, y_pred = self.val_loss(model, test_dl)
        return y_true, y_pred

    def evaluate(self):
        X, Y = self.preprocess()
        X_train, X_test, y_train, y_test = self.split_data(X, Y)
        emb_sizes = self.get_emb_sizes(X)

        model = ContEmbModel(embedding_sizes=emb_sizes,
                             n_cont=len(self.cont_col),
                             target_dim=len(pd.Series(Y).unique())
                             )

        print(to_device(model, DEVICE))

        train_ds = ContEmbDataset(X_train, y_train, self.cat_col, self.cont_col)
        test_ds = ContEmbDataset(X_test, y_test, self.cat_col, self.cont_col)

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=True)

        y_true, y_pred = self.train_loop(train_dl, test_dl, model, epochs=8, lr=0.05, wd=0.00001)
        print(classification_report(y_true, y_pred))
        return y_true, y_pred

    def train(self):
        X, Y = self.preprocess()
        emb_sizes = self.get_emb_sizes(X)

        model = ContEmbModel(embedding_sizes=emb_sizes,
                             n_cont=len(self.cont_col),
                             target_dim=len(pd.Series(Y).unique())
                             )

        print(to_device(model, DEVICE))

        train = ContEmbDataset(X, Y, self.cat_col, self.cont_col)
        train_dl = DataLoader(train, batch_size=self.batch_size, shuffle=True)

        def train_loop(train_dl, model, epochs, lr=0.01, wd=0.0):
            optim = self.get_optimizer(model, lr=lr, wd=wd)
            for epoch in range(epochs):
                loss, weights = self.train_model(model, optim, train_dl)
                print("training loss: ", loss)
            return weights

        weights = train_loop(train_dl, model, epochs=8, lr=0.05, wd=0.00001)
        return weights
