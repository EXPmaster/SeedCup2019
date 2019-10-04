import pandas as pd
import numpy as np
import xgboost as xgb


def pred_blank(trainset, tset, name):
    feature = ['plat_form', 'biz_type', 'product_id', 'cate1_id', 'cate2_id',
               'cate3_id', 'seller_uid', 'company_name']
    train = np.array(trainset[feature])
    train_target = list(trainset[name])
    test = np.array(tset[feature])
    # val_target = list(tset[name])
    label_set = list(set(train_target))
    count_train_target = len(label_set)
    label_dict = {key: value for value, key in enumerate(label_set)}

    train_target_idx = []
    for item in train_target:
        train_target_idx.append(label_dict[item])
    train_target_idx = np.array(train_target_idx)

    dtrain = xgb.DMatrix(train, label=train_target_idx)
    del train
    param1 = {
        'booster': 'gbtree',
        'colsample_bytree': 0.8,
        'eta': 0.1,
        'max_depth': 10,
        'num_class': count_train_target,
        'objective': 'multi:softmax',
        'gamma': 0.2,
        # 'subsample': 0.8,
        'min_child_weight': 2
        # 'tree_method': 'gpu_hist'
    }
    num_round = 10
    gbt = xgb.train(param1, dtrain, num_round, evals=[(dtrain, name)])
    del dtrain
    dtest = xgb.DMatrix(test)
    pred_idx = gbt.predict(dtest)
    del dtest
    new_dict = {v: k for k, v in label_dict.items()}
    predicts = [new_dict[item] for item in pred_idx]
    tset[name].loc[:] = predicts[:]

    return tset[name]