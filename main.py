import xgboost as xgb
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from DataLoader import handle_uid
from evaluation import calculateAllMetrics
from sklearn.datasets import dump_svmlight_file
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import re

pd.set_option('display.max_columns', None)

data_file = './train_data.csv'
test_file = './SeedCup2019_pre/SeedCup_pre_test.csv'
submit_file = './submit.txt'
read_rows = 120000

# TODO 地理特征处理
def time_predict(pay_time, total):
    # pay_time为支付时间，total为小时数
    i = 0
    date = []
    for item in pay_time:
        delta = datetime.timedelta(days=total[i] // 24, hours=total[i] % 24)
        new_date = item + delta
        if new_date.hour <= 6:
            new_date += datetime.timedelta(days=0, hours=10)
        date.append(new_date)
        i += 1
    return date  # 返回新世间


def fill_blank(dataset, validset, col_from, col_to):
    print('filling {} blanks...'.format(col_to))
    mosts = dataset[col_to].value_counts().index[0]
    array = []
    for item in validset[col_from]:
        try:
            idx = dataset[dataset[col_from] == item].index.tolist()[0]
        except IndexError:
            array.append(mosts)
        else:
            array.append(dataset[col_to].loc[idx])
    print('finished')
    validset[col_to].loc[:] = array[:]
    # print(validset[col_to])
    return validset[col_to]


def gen_dataset(dataset, validset, mode='valid'):
    # 下单时间
    train_begin_time = pd.to_datetime(dataset['payed_time'])
    valid_begin_time = pd.to_datetime(validset['payed_time'])
    # 训练集收货时间
    train_signed_time = pd.to_datetime(dataset['signed_time'])
    # 验证集收货时间
    if mode == 'valid':
        valid_signed_time = pd.to_datetime(validset['signed_time'])
    len_train = len(dataset)
    # 填补特征

    feature = ['lgst_company', 'warehouse_id', 'shipped_prov_id',
    'shipped_city_id']
    for item in feature:
        validset[item] = np.nan

    # validset['warehouse_id'] = fill_blank(dataset, validset, 'seller_uid', 'warehouse_id')
    validset['shipped_prov_id'] = fill_blank(dataset, validset, 'seller_uid', 'shipped_prov_id')
    validset['shipped_city_id'] = fill_blank(dataset, validset, 'seller_uid', 'shipped_city_id')

    # 构造长据集
    dataset = pd.concat([dataset, validset], ignore_index=True)

    # 填补缺失值
    for item in dataset.keys():
        dataset[item] = dataset[item].fillna(dataset[item].value_counts().index[0])
        # dataset[item] = dataset[item].fillna(-9999)
    # 归一化时间
    """
    pay_ship_norm = (dataset['pay_ship'] - dataset['pay_ship'].min()) /\
                 (dataset['pay_ship'].max() - dataset['pay_ship'].min())
    ship_get_norm = (dataset['ship_get'] - dataset['ship_get'].min()) /\
                 (dataset['ship_get'].max() - dataset['ship_get'].min())
    get_dlv_norm = (dataset['get_dlv'] - dataset['get_dlv'].min()) /\
                 (dataset['get_dlv'].max() - dataset['get_dlv'].min())
    dlv_sign_norm = (dataset['dlv_sign'] - dataset['dlv_sign'].min()) /\
                 (dataset['dlv_sign'].max() - dataset['dlv_sign'].min())
    """
    # print(dataset)
    # 处理id
    product_id = handle_uid(dataset['product_id'])
    product_id_dummy = pd.get_dummies(product_id, prefix=product_id)
    """
    
    seller_uid = (dataset['seller_uid'] - dataset['seller_uid'].min()) /\
                 (dataset['seller_uid'].max() - dataset['seller_uid'].min())
    """
    # one-hot
    # 交易平台
    plat_form_dummy = pd.get_dummies(
        dataset['plat_form'], prefix=dataset[['plat_form']].columns[0])
    # 业务来源
    biz_type_dummy = pd.get_dummies(
        dataset['biz_type'], prefix=dataset[['biz_type']].columns[0])
    # 订单创建时间
    # creat_time = pd.to_datetime(dataset['create_time'])

    # 一级类目
    cate1_id_dummy = pd.get_dummies(
        dataset['cate1_id'], prefix=dataset[['cate1_id']].columns[0])
    # 二级类目
    cate2_id_dummy = pd.get_dummies(
        dataset['cate2_id'], prefix=dataset[['cate2_id']].columns[0])
    # 三级类目
    cate3_id_dummy = pd.get_dummies(
        dataset['cate3_id'], prefix=dataset[['cate3_id']].columns[0])
    # TODO 商家公司id, pre_selling, 买家uid 未使用
    # 商家id
    seller_uid_dummy = pd.get_dummies(
        dataset['seller_uid'], prefix=dataset[['seller_uid']].columns[0])
    # 物流公司id
    lgst_company_dummy = pd.get_dummies(
        dataset['lgst_company'], prefix=dataset[['lgst_company']].columns[0])
    # 仓库id
    warehouse_id_dummy = pd.get_dummies(
        dataset['warehouse_id'], prefix=dataset[['warehouse_id']].columns[0])
    # 发货省份id
    shipped_prov_id_dummy = pd.get_dummies(
        dataset['shipped_prov_id'], prefix=dataset[['shipped_prov_id']].columns[0])
    # 发货城市id
    shipped_city_id_dummy = pd.get_dummies(
        dataset['shipped_city_id'], prefix=dataset[['shipped_city_id']].columns[0])
    # 收货省份id
    rvcr_prov_name_dummy = pd.get_dummies(
        dataset['rvcr_prov_name'], prefix=dataset[['rvcr_prov_name']].columns[0])
    # 收货城市id
    rvcr_city_name_dummy = pd.get_dummies(
        dataset['rvcr_city_name'], prefix=dataset[['rvcr_city_name']].columns[0])
    frames = [plat_form_dummy, biz_type_dummy, product_id_dummy, cate1_id_dummy, cate2_id_dummy,
              cate3_id_dummy, seller_uid_dummy, lgst_company_dummy, warehouse_id_dummy,
              shipped_prov_id_dummy, shipped_city_id_dummy, rvcr_prov_name_dummy,
              rvcr_city_name_dummy]
    new_dataset = pd.concat(frames, axis=1)

    train_set = new_dataset[:len_train]
    valid_set = new_dataset[len_train:]
    if mode == 'valid':
        return train_set, valid_set, train_begin_time,\
               valid_begin_time, train_signed_time, valid_signed_time
    else:
        return train_set, valid_set, train_begin_time, valid_begin_time, train_signed_time


def load_data(mode='valid', label='total_time'):
    print('loading dataset...')
    train_data = pd.read_csv('train_data.csv', nrows=read_rows)

    # 可视化
    """
    for item in train_data.keys():
        if not re.match('[a-z]*_time', item) and item != 'preselling_shipped_time':
            sns.distplot(train_data[item])
            print(item + ' finished')
            plt.show()
          
    sns.distplot(train_data['product_id'])
    plt.show()
    sns.distplot(train_data['seller_uid'])
    plt.show()
    
    sns.regplot(x='seller_uid', y='shipped_city_id', data=train_data)
    plt.show()
    sns.regplot(x='seller_uid', y='shipped_prov_id', data=train_data)
    plt.show()
    sns.regplot(x='total_time', y='shipped_city_id', data=train_data)
    plt.show()
    sns.regplot(x='total_time', y='shipped_prov_id', data=train_data)
    plt.show()
    sns.regplot(x='rvcr_prov_name', y='total_time', data=train_data)
    plt.show()
    """
    target = train_data[label]
    train_data = train_data.drop([label], axis=1)
    if mode == 'valid':
        train_set, valid_set, train_target, valid_target = train_test_split(
            train_data, target, test_size=0.5)

        train_set, valid_set, train_begin_time, valid_begin_time, \
        train_signed_time, valid_signed_time = gen_dataset(train_set, valid_set, mode='valid')
        print('load finished')
        return np.array(train_set), np.array(valid_set), np.array(train_target), \
               np.array(valid_target), train_begin_time, \
               train_signed_time, valid_begin_time, valid_signed_time

    else:
        # train_data.sample(frac=0.66, replace=True, random_state=None)
        valid_set = pd.read_csv(test_file, sep='\t')
        train_set, valid_set, train_begin_time, valid_begin_time, \
        train_signed_time = gen_dataset(train_data, valid_set, mode='test')
        print('load finished')
        return np.array(train_set), np.array(valid_set), np.array(target), \
               train_begin_time, train_signed_time, valid_begin_time


def my_loss_fun(y_pred, y_true):
    n = len(y_pred)
    y_true = y_true.get_label()
    # loss = ((y_pred - y_true) ** 2).mean()
    grad = y_pred - y_true
    hess = np.power(np.abs(grad), 0.5)

    """
    if np.sum(y_pred) > np.sum(y_true):
        loss = np.sqrt(((y_pred - y_true) ** 2).mean()) * 10
        grad = 10 * (y_pred - y_true) * (n * np.sum((y_pred - y_true) ** 2)) ** (-0.5)
        hess = 10 * y_pred * (n * np.sum(((y_pred - y_true) ** 2))) ** (-0.5) - n ** (-0.5) *\
               (np.sum(((y_pred - y_true) ** 2))) ** (-1.5) * (y_pred - y_true) ** 2
    else:
        loss = np.sqrt(((y_pred - y_true) ** 2).mean())
        grad = (y_pred - y_true) * (n * np.sum((y_pred - y_true) ** 2)) ** (-0.5)
        hess = y_pred * (n * np.sum(((y_pred - y_true) ** 2))) ** (-0.5) - n ** (-0.5) * \
               (np.sum(((y_pred - y_true) ** 2))) ** (-1.5) * (y_pred - y_true) ** 2
               """
    return grad, hess


def train(train_set, train_target, valid_set, valid_target=None):
    print('training...')
    dtrain = xgb.DMatrix(train_set, label=train_target)
    del train_set, train_target
    param = {
        'booster': 'gbtree',
        'colsample_bytree': 0.8,
        'eta': 0.1,
        'max_depth': 3000,    # 1000
        'objective': 'reg:squarederror',
        'gamma': 0.2,
        # 'subsample': 1.0,
        'min_child_weight': 30

        # 'tree_method': 'gpu_hist'
    }
    num_round = 9  # 8
    
    bst = xgb.train(param, dtrain, num_round)
    print('train finished')

    del dtrain
    dvalid = xgb.DMatrix(valid_set, label=valid_target)
    # dvalid = xgb.DMatrix(valid_set)
    # make prediction
    predict_total_hours = bst.predict(dvalid)
    predict_date = time_predict(valid_begin_time, predict_total_hours)

    otp, rs = calculateAllMetrics(valid_signed_time, predict_date)
    print('on time percent: %lf\nrank score: %lf' % (otp, rs))

    return predict_date


def submit(date):
    print('writing to file...')
    with open(submit_file, 'w') as f:
        for i in range(len(date)):
            frmt = str(date[i].year) + '-' + str(date[i].month) +\
                  '-' + str(date[i].day)
            f.write(frmt + ' ' + str(date[i].hour) + '\n')
    print('finished')


if __name__ == '__main__':
    # load_data()

    train_set, valid_set, train_target, valid_target, train_begin_time, \
    train_signed_time, valid_begin_time, valid_signed_time = load_data(mode='valid')
    
    """
    train_set, valid_set, train_target, train_begin_time, \
    train_signed_time, valid_begin_time = load_data(mode='test')
    """
    date = train(train_set, train_target, valid_set, valid_target)
    # date = train(train_set, train_target, valid_set)

    # submit(date)
