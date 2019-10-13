import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

train_file = './data/SeedCup_final_train.csv'
test_file = './data/SeedCup_final_test.csv'
pd.set_option('display.max_columns', None)


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
            dataset[date_str][i] = dataset[date_str][i].replace(
                '-99', str(np.nan))
    print('replace ' + date_str + ' finished')


def showfreq(time):
    d_freq = {}
    for date in time:
        if date not in d_freq:
            d_freq[date] = 1
        else:
            d_freq[date] += 1
    print(sorted(d_freq.items(), key=lambda item: item[1]))


def getfreq(time):
    d_freq = {}
    for date in time:
        if date not in d_freq:
            d_freq[date] = 1
        else:
            d_freq[date] += 1
    return dict(sorted(d_freq.items(), key=lambda item: item[1], reverse=True))


def handle_uid(uid):
    usr_id_idx = []
    usr_freq = getfreq(uid)
    for item in uid.values:
        usr_id_idx.append(usr_freq[item])
    return usr_id_idx


def handle_switch(first_time, second_time):
    if (first_time - second_time).days * 24 < 0:
        time = np.nan
    else:
        time = (first_time - second_time).days * 24 + (first_time -
                                                       second_time).seconds // 3600
    # if time > 400:
        # time = 400
    return time


def get_time_diff():
    print('getting time diff...')
    pay_ship_diff = []
    ship_get_diff = []
    get_dlv_diff = []
    dlv_sign_diff = []
    day_in_total = []
    for i in range(len(dataset['got_time'])):
        pay_ship_diff.append(handle_switch(shipped_time[i], payed_time[i]))
        ship_get_diff.append(handle_switch(got_time[i], shipped_time[i]))
        get_dlv_diff.append(handle_switch(dlved_time[i], got_time[i]))
        dlv_sign_diff.append(handle_switch(signed_time[i], dlved_time[i]))

        day_in_total.append(handle_switch(signed_time[i], payed_time[i]))

    times = {
        'pay_ship': pay_ship_diff,
        'ship_get': ship_get_diff,
        'get_dlv': get_dlv_diff,
        'dlv_sign': dlv_sign_diff,
        'total_time': day_in_total
    }
    time_diff_data = pd.DataFrame(times)
    print('finished')
    #print(time_diff_data)
    return time_diff_data


if __name__ == '__main__':
    print('Loading files...')
    dataset = pd.read_csv(train_file, sep='\t')
    dataset.drop(['preselling_shipped_time'], axis=1, inplace=True)
    # 修复时间的错误值
    fix_time('got_time')
    fix_time('dlved_time')
    dataset.dropna(axis=0, how='any', inplace=True)
    # 发货时间
    shipped_time = pd.to_datetime(dataset['shipped_time'])
    # 揽件时间
    got_time = pd.to_datetime(dataset['got_time'])
    # 走件时间
    dlved_time = pd.to_datetime(dataset['dlved_time'])
    # 订单支付时间
    payed_time = pd.to_datetime(dataset['payed_time'])
    # 签收时间
    signed_time = pd.to_datetime(dataset['signed_time'])
    # 时间差
    time_diff = get_time_diff()

    # TODO 商家公司频率
    frames = [dataset, time_diff]
    new_dataset = pd.concat(frames, axis=1)
    new_dataset.dropna(axis=0, how='any', inplace=True)
    del dataset
    # target = np.array(new_dataset['total_time'])
    # train_data = np.array(new_dataset.drop(['total_time'], axis=1))
    print('load data finished')
    print('saving dataset...')
    new_dataset.to_csv('train_data.csv')
    print('data saved successfully')
    # 可视化
    """
    for item in dataset.keys():
        if not re.match('[a-z]*_time', item) and item != 'preselling_shipped_time':
            sns.distplot(dataset[item])
            print(item + ' finished')
            plt.show()
    """