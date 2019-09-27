import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


pd.set_option('display.max_columns', None)

data_file = './train_data.csv'
test_file = './SeedCup2019_pre/SeedCup_pre_test.csv'
submit_file = './submit.txt'
read_rows = 3000




def fill_time(date_str):
    # 填补时间无效值0
    a = []
    # df = pd.DataFrame()
    pi = r'\d+-\d+-\d+ \d+:\d+:\d+'
    pa = re.compile(pi)
    print('fill_time...')
    for i in range(len(dataset[date_str])):
        key = dataset[date_str][i]
        # print(train_reader['got_time'][i])
        # print(pa.findall(key))
        if pa.findall(key) == a:
            # print(train_reader['dlved_time'][i],i)
            dataset[date_str][i] = dataset[date_str][i].replace(
                    '0', '1970-01-01 08:00:00')
    print('replace ' + date_str + ' finished')


def showfreq(time):
    d_freq = {}
    for date in time:
        if date not in d_freq:
            d_freq[date] = 1
        else:
            d_freq[date] += 1
    print(sorted(d_freq.items(), key=lambda item: item[1]))
    print(d_freq)


def plotfreq(hours):
    p_freq={}
    for item in hours:
     date = ((item.second) // 3600 + item.minute // 60 + item.hour)
     if date not in p_freq:
         p_freq[date] = 1
     else:
         p_freq[date] += 1
    a=list(range(24))
    b=[]
    for i in range(24):
        b.append(p_freq[i])
    plt.bar(range(len(a)), b, width=0.3)
    plt.xticks(range(len(a)), a, rotation=90)
    plt.grid(alpha=0.3)
    plt.show()
    plt.savefig('1.svg')

if __name__ == '__main__':
    train_data = pd.read_csv('train_data.csv')
    #fill_time('preselling_shipped_time')
    #预售时间
    #preselling_shipped_time = pd.to_datetime(dataset['preselling_shipped_time'])
    #创建时间
    #create_time=pd.to_datetime(dataset['create_time'])
    #签收时间
    signed_time = pd.to_datetime(train_data['signed_time'])
    #支付时间
    payed_time=pd.to_datetime(train_data['payed_time'])
    #卖家ID
    seller_uid=pd.to_datetime(train_data['seller_uid'])
    #发货城市ID
    shipped_city_id = pd.to_datetime(train_data['shipped_city_id'])
    #总时间
    total_time=train_data['total_time']
    '''sns.set(color_codes=True)
    np.random.seed(sum(map(ord, "regression")))'''

    sns.regplot(x="plat_form", y="warehouse_id", data=train_data)
    plt.show()
    '''
    sns.set(color_codes=True)
    np.random.seed(sum(map(ord, "regression")))
    tips = sns.load_dataset("tips")
    print(tips.head())
    sns.regplot(x="total_bill", y="tip", data=tips)
    '''
    '''
    plotfreq(payed_time)
    '''