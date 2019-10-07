import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.backends.cudnn as cudnn
import pandas as pd
"""
没卵用
"""


class EncodeData(Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset[['shipped_prov_id', 'shipped_city_id',
                                'rvcr_prov_name', 'rvcr_city_name']]
        self.target = target
        self.dataset = np.array(self.dataset)
        # mean = self.dataset.mean(axis=0)
        # std = self.dataset.std(axis=0)
        # self.dataset = (self.dataset - mean) / std

    def __getitem__(self, item):
        data = self.dataset[item]
        target = self.target[item]
        return data, target

    def __len__(self):
        return len(self.dataset)


class EncodePred(Dataset):
    def __init__(self, tset):
        self.tset = tset[['shipped_prov_id', 'shipped_city_id',
                                'rvcr_prov_name', 'rvcr_city_name']]
        self.tset = np.array(self.tset)
        # mean = self.tset.mean(axis=0)
        # std = self.tset.std(axis=0)
        # self.tset = (self.tset - mean) / std

    def __getitem__(self, item):
        return self.tset[item]

    def __len__(self):
        return len(self.tset)


class AutoEncoder(nn.Module):
    def __init__(self, n1, n2, n3, n4):
        self.embed_dim = 1024
        super().__init__()
        self.embed_ship_prov = nn.Embedding(n1 + 1, self.embed_dim)
        self.embed_ship_city = nn.Embedding(n2 + 1, self.embed_dim)
        self.embed_rvcr_prov = nn.Embedding(n3 + 1, self.embed_dim)
        self.embed_rvcr_city = nn.Embedding(n4 + 1, self.embed_dim)

        self.encoder = nn.Sequential(
            nn.Linear(4 * self.embed_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),

            nn.Linear(2, 1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1)
        )

    def forward(self, x):
        embed_ship_prov = self.embed_ship_prov(x[:, 0].long())
        embed_ship_city = self.embed_ship_city(x[:, 1].long())
        embed_rvcr_prov = self.embed_rvcr_prov(x[:, 2].long())
        embed_rvcr_city = self.embed_rvcr_city(x[:, 3].long())
        x = torch.cat((embed_ship_prov, embed_ship_city, embed_rvcr_prov,
                       embed_rvcr_city), 1)

        encode = self.encoder(x)
        decode = self.decoder(encode)
        encode = encode.reshape(encode.size(0))
        decode = decode.reshape(decode.size(0))

        return encode, decode


def predict_distance(trainset, testset, target):
    def train(epoch, least_loss):
        autoencoder.train()
        for idx, (traindata, target) in enumerate(trainloader):
            if use_cuda:
                traindata = traindata.cuda()
                target = target.cuda()
            traindata = traindata.float()
            target = target.float()
            optimizer.zero_grad()
            encode, decode = autoencoder(traindata)

            loss = criterion(decode, target)
            if idx == 1:
                print("==> epoch {}: loss is {} ".format(epoch, loss))
            if loss < least_loss:
                least_loss = loss
                print('loss: {}, saving model...'.format(loss))
                torch.save(autoencoder, './model.pkl')

            loss.backward()
            optimizer.step()

        return least_loss

    def predict():
        autoencoder = torch.load('./model.pkl')
        autoencoder.eval()
        pred_distance = []
        for idx, data in enumerate(testLoader):
            if use_cuda:
                data = data.cuda()
            data = data.float()
            output, decode = autoencoder(data)
            if use_cuda:
                output = output.cpu()
            for item in output.detach().numpy():
                pred_distance.append(abs(item))
        return pred_distance

    print('encoding...')
    batchSize = 32
    total_epoch = 20
    LR = 1e-3

    testset = pd.concat([trainset, testset], ignore_index=True)
    n1 = testset['shipped_prov_id'].max()
    n2 = testset['shipped_city_id'].max()
    n3 = testset['rvcr_prov_name'].max()
    n4 = testset['rvcr_city_name'].max()

    dataset = EncodeData(trainset, target)
    testset = EncodePred(testset)
    trainloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testset, batch_size=batchSize, shuffle=False)

    autoencoder = AutoEncoder(n1, n2, n3, n4)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    criterion = nn.MSELoss()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        cudnn.benchmark = True
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    autoencoder.to(device)

    least_loss = 1000
    for i in range(total_epoch):
        least_loss = train(i, least_loss)
    predicts = predict()
    # print(predicts)
    print('encode finished')
    return pd.DataFrame(predicts)


