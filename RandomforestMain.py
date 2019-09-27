import xgboost as xgb
import numpy as np
import pandas as pd
import datetime
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.metrics import fbeta_score, make_scorer
pd.set_option('display.max_columns', None)

data_file = './train_data.csv'
test_file = './SeedCup2019_pre/SeedCup_pre_test.csv'
submit_file = './submit.txt'
read_rows = 100000


def time_predict(pay_time, total):
    # pay_time为支付时间，total为小时数
    i = 0
    date = []
    hour = []
    for item in pay_time:
        delta = datetime.timedelta(days=total[i] // 24, hours=total[i] % 24)
        new_date = item + delta
        date.append(new_date)
        hour.append(new_date.hour)
        i += 1
    return date, hour  # 返回新世间


def onTimePercent(pred_date, real_date):
    real_date = list(real_date)
    total = len(pred_date)
    count = 0
    for i in range(total):
        if ((pred_date[i] - real_date[i]).days) <= 0:
            count += 1
    print('on time percent: %lf' % (count / total))


def rankScore(real_signed_time, pred_signed_time):
    rst = []
    for item in real_signed_time:
        rst.append(item.hour)
    print('MSE: %lf' % mean_squared_error(rst, pred_signed_time))


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
    'shipped_city_id', 'pay_ship', 'ship_get',
    'get_dlv', 'dlv_sign']
    for item in feature:
        validset[item] = np.nan
    # 构造长据集
    dataset = pd.concat([dataset, validset], ignore_index=True)

    # 填补缺失值
    for item in dataset.keys():
        dataset[item] = dataset[item].fillna(dataset[item].value_counts().index[0])

    # print(dataset)

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
    # TODO 商家id, 商家公司id, product_id, pre_selling 未使用
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
    frames = [plat_form_dummy, biz_type_dummy, cate1_id_dummy, cate2_id_dummy,
              cate3_id_dummy, lgst_company_dummy, warehouse_id_dummy,
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
    target = train_data[label]
    train_data = train_data.drop([label], axis=1)
    if mode == 'valid':
        train_set, valid_set, train_target, valid_target = train_test_split(
            train_data, target, test_size=0.33)

        train_set, valid_set, train_begin_time, valid_begin_time, \
        train_signed_time, valid_signed_time = gen_dataset(train_set, valid_set, mode='valid')
        print('load finished')
        return np.array(train_set), np.array(valid_set), np.array(train_target), \
               np.array(valid_target), train_begin_time, \
               train_signed_time, valid_begin_time, valid_signed_time

    else:
        valid_set = pd.read_csv(test_file, sep='\t')
        train_set, valid_set, train_begin_time, valid_begin_time, \
        train_signed_time = gen_dataset(train_data, valid_set, mode='test')
        print('load finished')
        return np.array(train_set), np.array(valid_set), np.array(target), \
               train_begin_time, train_signed_time, valid_begin_time

"""def scorer(pred_date, real_date,pred_signed_time):
    real_signed_time=real_date
    real_date = list(real_date)
    total = len(pred_date)
    count = 0
    for i in range(total):
        if ((pred_date[i] - real_date[i]).days) <= 0:
            count += 1
    (count / total)
    rst = []
    for item in real_signed_time:
        rst.append(item.hour)
    return 100*(count/total)-mean_squared_error(rst, pred_signed_time)"""


"""def my_Mse(real_signed_time, pred_signed_time):
     rst = []
     for item in real_signed_time:
         rst.append(item.hour)
     return mean_squared_error(rst, pred_signed_time)

myscore = make_scorer(my_Mse, greater_is_better=False)"""


"""ftwo_scorer  =  make_scorer(fbeta_score,beta = 2)"""
def train():
    print('training...')
    #   param = {'n_estimators': 20, 'criterion ': 'mse'}
    #   num_round = 5
    #   param_grid = {'min_weight_fraction_leaf': np.arange(0,0.1,0.1)}
    rdf=RandomForestRegressor(n_estimators=15,criterion = 'mse',max_depth =3)

    rdf = rdf.fit(train_set,train_target)
    print('train finished')
    # make prediction
    predict_total_hours = rdf.predict(valid_set)
    predict_date, predict_hour = time_predict(valid_begin_time, predict_total_hours)

    onTimePercent(predict_date, valid_signed_time)
    rankScore(valid_signed_time, predict_hour)

    return predict_date, predict_hour


def submit(date, hour):
    print('writing to file...')
    with open(submit_file, 'w') as f:
        for i in range(len(date)):
            frmt = str(date[i].year) + '-' + str(date[i].month) +\
                  '-' + str(date[i].day)
            f.write(frmt + ' ' + str(hour[i]) + '\n')
    print('finished')


if __name__ == '__main__':

    train_set, valid_set, train_target, valid_target, train_begin_time, \
    train_signed_time, valid_begin_time, valid_signed_time = load_data(mode='valid')
    """
    train_set, valid_set, train_target, train_begin_time, \
    train_signed_time, valid_begin_time = load_data(mode='test')
    """
    print(train_set.shape)
    dtrain = xgb.DMatrix(train_set, label=train_target)
    dvalid = xgb.DMatrix(valid_set, label=valid_target)
    # dvalid = xgb.DMatrix(valid_set)

    date, hour = train()
    submit(date, hour)