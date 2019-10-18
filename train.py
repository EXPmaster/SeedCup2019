from main import load_data, time_predict
from evaluation import calculateAllMetrics
from newDataLoader import Trainset, Validset
from Config import Config
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from Network import My_MSE_loss, Network
import os
from tqdm import tqdm
import datetime

opt = Config()

train_set, valid_set, train_target, valid_target, train_begin_time, \
train_signed_time, valid_begin_time, valid_signed_time = load_data(mode='valid')
trainset = Trainset(train_set, train_target)
validset = Validset(valid_set, valid_begin_time, valid_signed_time)
trainloader = DataLoader(trainset, batch_size=opt.TRAIN_BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=opt.VAL_BATCH_SIZE, shuffle=False)


# criterion = My_MSE_loss.apply
criterion = nn.MSELoss()
net = Network(opt)
if opt.USE_CUDA:
    print("==> using CUDA")
    """
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count())).cuda()
    """
    device = 'cuda:0'
    net.to(device)
    cudnn.benchmark = True
optimizer = torch.optim.Adam(net.parameters(), lr=opt.LR)
threshold_rs = 40
threshold_otp = 0.98


def train(epoch):
    net.train()
    for idx, (traindata, target) in enumerate(tqdm(trainloader)):
        if opt.USE_CUDA:
            traindata = traindata.cuda()
            target = target.cuda()
        traindata = traindata.float()
        target = target.float()
        optimizer.zero_grad()
        outputs = net(traindata)
        # target = target.argmax(dim=1)
        outputs = outputs.reshape(outputs.size(0))
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()
        if idx == 1:
            print("==> epoch {}: loss is {} ".format(epoch, loss.data))


@torch.no_grad()
def val():
    net.eval()
    global threshold_rs, threshold_otp
    print('validating...')
    predict_total_hours = []
    valid_begin_time = []
    valid_signed_time = []
    for idx, (validdata, begin, target) in enumerate(tqdm(validloader)):
        if opt.USE_CUDA:
            validdata = validdata.cuda()
        validdata = validdata.float()
        outputs = net(validdata)
        if opt.USE_CUDA:
            outputs = outputs.cpu()
        outputs = outputs.numpy()
        for i in range(len(validdata)):
            predict_total_hours.append(outputs[i][0])
            payed_time = datetime.datetime.strptime(
                begin[i], "%Y-%m-%d %H:%M:%S")
            signed_time = datetime.datetime.strptime(
                target[i], "%Y-%m-%d %H:%M:%S")
            valid_begin_time.append(payed_time)
            valid_signed_time.append(signed_time)

    predict_date = time_predict(valid_begin_time, predict_total_hours)
    otp, rs = calculateAllMetrics(valid_signed_time, predict_date)
    print('on time percent: %lf\nrank score: %lf' % (otp, rs))

    if rs < threshold_rs and otp >= threshold_otp:
        print("==> saving model")
        if not os.path.exists('models'):
            os.mkdir('models')
        threshold_rs = rs
        threshold_otp = otp
        torch.save(net, './models/model_{}_{}.pt'.format(otp, rs))


if __name__ == '__main__':
    print('training...')
    for i in range(opt.NUM_EPOCHS):
        train(i + 1)
        val()
    print('train finished')





