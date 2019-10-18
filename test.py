from main import load_data, submit, time_predict
from newDataLoader import Trainset, Testset
from Config import Config
import torch
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import datetime

train_set, valid_set, train_target, train_begin_time, \
    train_signed_time, test_begin_time = load_data(mode='test')

opt = Config()
trainset = Trainset(train_set, train_target)
testset = Testset(valid_set, test_begin_time)
testloader = DataLoader(testset, batch_size=opt.TEST_BATCH_SIZE, shuffle=False)


@torch.no_grad()
def test():
    net = torch.load('model.pt')
    if opt.USE_CUDA:
        print("==> using CUDA")
        """
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count())).cuda()
        """
        device = 'cuda:0'
        net.to(device)
        cudnn.benchmark = True
    net.eval()
    print('testing...')
    predict_total_hours = []
    test_begin_time = []
    test_signed_time = []
    for idx, (testdata, begin, target) in enumerate(testloader):
        if opt.USE_CUDA:
            testdata = testdata.cuda()
        testdata = testdata.float()
        outputs = net(testdata)
        if opt.USE_CUDA:
            outputs = outputs.cpu()
        outputs = outputs.numpy()
        for i in range(len(testdata)):
            predict_total_hours.append(outputs[i][0])
            payed_time = datetime.datetime.strptime(
                begin[i], "%Y-%m-%d %H:%M:%S")
            signed_time = datetime.datetime.strptime(
                target[i], "%Y-%m-%d %H:%M:%S")
            test_begin_time.append(payed_time)
            test_signed_time.append(signed_time)

    predict_date = time_predict(test_begin_time, predict_total_hours)
    return predict_date


if __name__ == '__main__':
    date = test()
    submit(date)

