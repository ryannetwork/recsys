import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import List


class EmbeddingDataset(Dataset):
    def __init__(self,
                 data,
                 cat_cols: List = None,
                 output_col: str = None):

        """
        class for creating Pytorch Dataset for model

        :param data - pd.DataFrame, input data for model
        :param cat_cols - list, names of cat columns
        :param output_col - str, label
        """

        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [col for col in data.columns
                          if col not in self.cat_cols + [output_col]]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.n, 1))

    def __len__(self):

        return self.n

    def __getitem__(self, idx):

        return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]


class EmbeddingNN(nn.Module):

    def __init__(self,
                 emb_dims: List,
                 no_of_cont: int,
                 lin_layer_sizes: List,
                 output_size: int,
                 emb_dropout: float,
                 lin_layer_dropouts: List,
                 model_class = 'regr'):

        """
        class for building experiments model

        :param emb_dims - list, size of embeddings for each cat column
        :param no_of_cont - int, number of cont variable
        :param lin_layer_sizes - list, size of lin layers
        :param output_size - int, final output
        :param emb_dropout - float, dropout prob for emb layer
        :param lin_layer_dropouts - list, dropout for lin layers
        """

        super().__init__()

        self.emb_dims = emb_dims
        self.model_class = model_class

        self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                         for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont,
                                    lin_layer_sizes[0])

        self.lin_layers = \
            nn.ModuleList([first_lin_layer] +
                          [nn.Linear(lin_layer_sizes[i],
                                     lin_layer_sizes[i + 1])
                           for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)

        nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size)
                                        for size in lin_layer_sizes])

        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size)
                                            for size in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):

        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i])
                 for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_cont_data], 1)
            else:
                x = normalized_cont_data

        for lin_layer, dropout_layer, bn_layer in \
                zip(self.lin_layers, self.droput_layers, self.bn_layers):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        if self.model_class == 'binary':
            x = F.sigmoid(x)

        return x

    def get_embeddings(self, cat_features: List,
                       emb_dims: List,
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


def train_embedding_model(model: nn.Module,
                          data: iter,
                          n_epochs: int,
                          criterion: nn.modules.loss,
                          optimizer: torch.optim) -> None:

    """
    :param model - class inherited from nn.Module with DL model
    :param data - iterator for batching data
    :param n_epochs - number of epochs
    :param criterion - loss for model
    :param optimizer - optimizer from torch for model
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(n_epochs):
        for y, cont_x, cat_x in data:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            y = y.to(device)

            preds = model(cont_x, cat_x)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'loss on epoch {epoch} is {loss}')
