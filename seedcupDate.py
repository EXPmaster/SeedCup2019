import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
import xgboost as xgb
import seaborn as sns
import re
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

train_file = 'SeedCup_pre_train.csv'
test_file = 'SeedCup_pre_test.csv'
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)   #解决print不完全打印

print('Loading files...')
dataset = pd.read_csv(train_file, sep='\t')
dataset = dataset[:2000]   #切片数据集
# 订单支付时间
payed_time = pd.to_datetime(dataset['payed_time'])
begin_time = payed_time
# 签收时间
signed_time = pd.to_datetime(dataset['signed_time'])


def fix_time(date_str):
    # 修复时间无效值
    a = []
    # df = pd.DataFrame()
    pi = r'\d+-\d+-\d+ \d+:\d+:\d+'
    pa = re.compile(pi)
    print('replacing...')
    for i in range(len(dataset[date_str])):
        key = dataset[date_str][i]
        # print(train_reader['got_time'][i])
        # print(pa.findall(key))
        if pa.findall(key) == a:
            # print(train_reader['dlved_time'][i],i)
            if date_str == 'dlved_time':
                dataset[date_str][i] = dataset[date_str][i].replace(
                    '-99', '2019-03-03 11:20:30')
            else:
                dataset[date_str][i] = dataset[date_str][i].replace(
                    '-99', '2019-03-02 17:37:05')
    print('replace ' + date_str + ' finished')


def showfreq(time):
    d_freq = {}
    for date in time:
        if date not in d_freq:
            d_freq[date] = 1
        else:
            d_freq[date] += 1
    print(sorted(d_freq.items(), key=lambda item: item[1]))


def get_time_diff():
    print('getting time diff...')
    pay_ship_diff = []
    ship_get_diff = []
    get_dlv_diff = []
    dlv_sign_diff = []
    time_in_total = []
    for i in range(len(dataset['got_time'])):
        pay_ship_diff.append((shipped_time[i] -
                              payed_time[i]).days *
                             24 +
                             (shipped_time[i] -
                              payed_time[i]).seconds //
                             3600)
        if ((got_time[i] - shipped_time[i]).days * 24 +
            (got_time[i] - shipped_time[i]).seconds // 3600) < 0:
            ship_get_diff.append((shipped_time[i] -
                                  got_time[i]).days *
                                 24 +
                                 (shipped_time[i] -
                                  got_time[i]).seconds //
                                 3600)
        else:
            ship_get_diff.append((got_time[i] -
                                  shipped_time[i]).days *
                                 24 +
                                 (got_time[i] -
                                  shipped_time[i]).seconds //
                                 3600)
        get_dlv_diff.append((dlved_time[i] - got_time[i]).days * 24 +
                            (dlved_time[i] - got_time[i]).seconds // 3600)
        dlv_sign_diff.append((signed_time[i] - dlved_time[i]).days *
                             24 + (signed_time[i] - dlved_time[i]).seconds // 3600)
        time_in_total.append(
            (signed_time[i] - payed_time[i]).days * 24 + (signed_time[i] - payed_time[i]).seconds // 3600)

    times = {
        'pay_ship': pay_ship_diff,
        'ship_get': ship_get_diff,
        'get_dlv': get_dlv_diff,
        'dlv_sign': dlv_sign_diff,
        'total_time': time_in_total
    }
    time_diff_data = pd.DataFrame(times)
    print('finished')
    return time_diff_data


if __name__ == '__main__':
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
    shipped_city_id = pd.get_dummies(
        dataset['shipped_city_id'], prefix=dataset[['shipped_city_id']].columns[0])
    # 收货省份id
    rvcr_prov_name_dummy = pd.get_dummies(
        dataset['rvcr_prov_name'], prefix=dataset[['rvcr_prov_name']].columns[0])
    # 收货城市id
    rvcr_city_name_dummy = pd.get_dummies(
        dataset['rvcr_city_name'], prefix=dataset[['rvcr_city_name']].columns[0])
    # 发货时间
    shipped_time = pd.to_datetime(dataset['shipped_time'])
    # 修复时间的错误值
    fix_time('got_time')
    fix_time('dlved_time')
    # 揽件时间
    got_time = pd.to_datetime(dataset['got_time'])
    # 走件时间
    dlved_time = pd.to_datetime(dataset['dlved_time'])

    # 时间差
    time_diff = get_time_diff()

    # TODO 商家公司频率
    frames = [plat_form_dummy, biz_type_dummy, cate1_id_dummy, cate2_id_dummy,
              cate3_id_dummy, lgst_company_dummy, warehouse_id_dummy,
              shipped_prov_id_dummy, rvcr_prov_name_dummy,
              rvcr_city_name_dummy, time_diff]
    new_dataset = pd.concat(frames, axis=1)
    del dataset
    target = np.array(new_dataset['total_time'])
    train_data = np.array(new_dataset.drop(['total_time'], axis=1))
    print('load data finished')
    print('saving dataset...')
    np.save('dataset.npy', train_data)
    np.save('target.npy', target)
    # 可视化
    """
    for item in dataset.keys():
        if not re.match('[a-z]*_time', item) and item != 'preselling_shipped_time':
            sns.distplot(dataset[item])
            print(item + ' finished')
            plt.show()
    """


tempdata=train_data[:2000]
temptarget=target[:2000]
print(train_data.size,train_data.shape,train_data.ndim)
print(tempdata.size,tempdata.shape,tempdata.ndim)
Xtrain,Xtest,Ytrain,Ytest=train_test_split(tempdata,temptarget,test_size=0.3)
clf=tree.DecisionTreeRegressor()
clf=clf.fit(Xtrain,Ytrain)
train_score=clf.score(Xtrain,Ytrain)
test_score = clf.score(Xtest, Ytest)
print(train_score,test_score)
#   print(clf.feature_importances_)


"""svr = SVR(kernel='rbf')
svr=svr.fit(Xtrain, Ytrain)
s_score=svr.score(Xtest, Ytest)
print(s_score)"""

def onTimePercent(pred_date, real_date):
    total = len(pred_date)
    count = 0
    print(total)
    for i in range(total):
        if (pred_date[i] - real_date[i]).days < 0:
            count += 1
    print('on time percent: %lf' % (count / total))
    print(count)

def rankScore(real_signed_time, pred_signed_time):
    print('MSE: %lf' % mean_squared_error(real_signed_time, pred_signed_time))

def time_predict(pay_time, total):
    # pay_time为支付时间，total为小时数
    for i in range(len(total)):
        delta = datetime.timedelta(days=total[i] // 24, hours=total[i] % 24)
        pay_time[i] = pay_time[i] + delta
    return pay_time  # 返回新世间
"""#    决策树
tempdata = train_data[:2000]
temptarget = target[:2000]  #取两千组实验
print(train_data.size, train_data.shape, train_data.ndim)
print(tempdata.size, tempdata.shape, tempdata.ndim)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(tempdata, temptarget, test_size=0.3)
dtrain = xgb.DMatrix(Xtrain, label=Ytrain)
dvalid = xgb.DMatrix(Xtest, label=Ytest)
clf = tree.DecisionTreeRegressor(criterion="mse")
clf = clf.fit(Xtrain, Ytrain)
train_score = clf.score(Xtrain, Ytrain)
test_score = clf.score(Xtest, Ytest)
print(train_score, test_score)
predict_hours = clf.predict(Xtest)
predict_time = time_predict(begin_time, predict_hours)
onTimePercent(predict_time, signed_time)
rankScore(predict_hours, Ytest)"""

#   随机森林
tempdata = train_data[:2000]
temptarget = target[:2000]  #取两千组实验
Xtrain, Xtest, Ytrain, Ytest = train_test_split(tempdata, temptarget, test_size=0.2)
print(Xtrain.shape,Ytest.shape)
dtrain = xgb.DMatrix(Xtrain, label=Ytrain)
dvalid = xgb.DMatrix(Xtest, label=Ytest)
clf = RandomForestRegressor()
clf = clf.fit(Xtrain, Ytrain)
train_score = clf.score(Xtrain, Ytrain)
test_score = clf.score(Xtest, Ytest)
print(train_score, test_score)
predict_hours = clf.predict(Xtest)
print('jjj',predict_hours.shape,begin_time.shape)
predict_time = time_predict(begin_time, predict_hours)
print('p_s',predict_time.shape,signed_time.shape)
onTimePercent(predict_time, signed_time)
rankScore(predict_hours, Ytest)
