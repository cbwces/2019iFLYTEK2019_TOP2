# coding: utf-8
import os
import time
from tqdm import tqdm
from numba import jit
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

signal_feature = ['转速信号1', '转速信号2', '压力信号1', '压力信号2', '温度信号', '流量信号', '电流信号']

def data_smoother(file_list, features):
    for file in file_list:
        df = pd.read_csv(file)
        for feature in features:
            df[feature+"_smooth"] = data_smooth(df[feature].values)
        df.to_csv(file, index=None)

@jit(nopython=True)
def data_smooth(x, alpha=20, beta=1):
    new_x = np.zeros(x.shape[0])
    new_x[0] = x[0]
    for i in range(1, len(x)):
        tmp = x[i-1] * (alpha - beta) / alpha + x[i] * beta / alpha
        new_x[i] = x[i] - tmp
    return new_x

def res(y_hat, data):
    data = data.get_label()
    temp = np.log(np.abs(y_hat + 1)) - np.log(np.abs(data + 1))
    rres = np.sqrt(np.dot(temp, temp) / len(temp))
    return 'res', rres, False

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False

n_samples = len(pd.read_csv('../data/shuishou/train/00fb58ecd675062e4423.csv'))
sample_duration = 0.02
sample_rate = n_samples * (1 / sample_duration)

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def high_pass_filter(x, low_cutoff=10000, sample_rate=sample_rate):
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = sosfilt(sos, x)
    return filtered_sig

def denoise_signal(x, wavelet='haar', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

def getfilelist(dir, filelist):
    newdir = dir
    if os.path.isfile(dir):
        filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newdir = os.path.join(dir, s)
            getfilelist(newdir, filelist)
    return filelist

def stat1(data,c,name):
    data = data.sort_values()[5:-5]
    c[name + '_max'] = data.max()
    c[name + '_min'] = data.min()
    c[name + '_mean'] = data.mean()
    c[name + '_ptp'] = data.ptp()
    c[name + '_std'] = data.std()
    c[name + '_median'] = data.median()
    c[name + '_kurt'] = data.kurt()
    c[name + '_mode'] = data.mode()
    c[name + '_mad'] = np.mean(np.abs(data - data.mean()))

    c[name + '_max_mode'] = c[name + '_max'] / c[name + '_mode']
    c[name + '_min_mode'] = np.abs(c[name + '_min'] / c[name + '_mode'])
    c[name + '_mean_mode'] = np.abs(c[name + '_mean'] / c[name + '_mode'])
    c[name + '_median_mode'] = np.abs(c[name + '_median'] / c[name + '_mode'])

    if ('压力信号1' in name) or ('压力信号2' in name):
        c[name + '_05'] = data.quantile(.05)
        c[name + '_25'] = data.quantile(.25)
        c[name + '_75'] = data.quantile(.75)
        c[name + '_95'] = data.quantile(.95)
    return c
    
def stat2(data,c,name):
    data = data.sort_values()[5:-5]
    c[name + '_max'] = data.max()
    c[name + '_min'] = data.min()
    c[name + '_mean'] = data.mean()
    c[name + '_ptp'] = data.ptp()
    c[name + '_std'] = data.std()
    c[name + '_median'] = data.median()
    c[name + '_kurt'] = data.kurt()#写成kurt
    return c

def process_sample_single(e,train_p):
    data = pd.read_csv(e)

    data['部件工作时长_diff'] = data['部件工作时长'].diff()
    data['累积量参数1_diff'] = data['累积量参数1'].diff()
    data['累积量参数2_diff'] = data['累积量参数2'].diff()
    data['开关1信号_diff'] = data['开关1信号'].diff()
    data['传动比'] = data['转速信号1'] / data['转速信号2']
    data['鸭梨比'] = data['压力信号1'] / data['压力信号2']
    data['传动比'] = data['传动比'].fillna(1)
    data['鸭梨比'] = data['鸭梨比'].fillna(1)
    data['传动比'] = data['传动比'].replace(np.inf, 1)
    data['传动比'] = data['传动比'].replace(-np.inf, 1)
    data['鸭梨比'] = data['鸭梨比'].replace(np.inf, 1)
    data['鸭梨比'] = data['鸭梨比'].replace(-np.inf, 1)
    lifemax = data['部件工作时长'].max()

    cmax=data.shape[0]
    data = data[:int(cmax*train_p)]

    c = {'train_file_name': os.path.basename(e)+str(train_p),
         '开关1_sum':data['开关1信号'].sum(),
         '告警1_sum':data['告警信号1'].sum(),
         '设备':data['设备类型'][0],
         'life':lifemax-data['部件工作时长'].values[-1]
         }
    for i in [
              '转速信号1',
              '转速信号2',
              '压力信号1',
              '压力信号2',
              '温度信号',
              '流量信号',
              '电流信号',
            ]:
            x_hp = high_pass_filter(data[i], low_cutoff=10000, sample_rate=sample_rate)
            x_dn = denoise_signal(x_hp, wavelet='haar', level=1)
            data[i] = data[i] - data[i].median()
            c=stat1(data[i],c,i)
            c=stat1(pd.Series(x_dn), c, i+"_wavelet")
            c=stat2(pd.Series(data_smooth(x_dn)), c, i+"_smooth")
            
    for i in [
              '部件工作时长', 
              '累积量参数1', 
              '累积量参数2',
              '传动比', 
              '鸭梨比', 
              '部件工作时长_diff',
            ]:
            c=stat2(data[i],c,i)
            
    this_tv_features = pd.DataFrame(c, index=[0])  
    return this_tv_features

def get_together(listp,istest,func):
    if istest:
        rst = pd.DataFrame()
        for e in tqdm(listp, ascii=True):
            rst = rst.append(func(e, 1))
    else:   
        train_p_list=[0.45, 0.55, 0.63, 0.68, 0.75, 0.85]
        rst = pd.DataFrame()
        for e in tqdm(listp, ascii=True):
            for train_p in train_p_list:
                rst = rst.append(func(e, train_p))
    return rst

def compute_loss(target, predict):
    temp = np.log(abs(target + 1)) - np.log(abs(predict + 1))
    res = np.sqrt(np.dot(temp, temp) / len(temp))
    return res

path = '../data/shuishou/'

train_list = getfilelist(path + 'train/', [])
test_list2 = getfilelist(path + 'test2/', [])

func=process_sample_single
train= get_together(train_list,False,func)
test2 = get_together(test_list2, True, func)

train['train_file_name'] = train['train_file_name'].apply(lambda x:x[:-4])
test2['train_file_name'] = test2['train_file_name'].apply(lambda x:x[:-1])

train = train.reset_index(drop=True)
test2 = test2.reset_index(drop=True)
train_shape = train.shape[0]
data=train.append(test2).reset_index(drop=True)
# data['shebei'] = data['设备']
lbe = LabelEncoder()
data['设备'] = lbe.fit_transform(data['设备'])
# alter_data=pd.get_dummies(data,columns=['设备'])

train = data[:train_shape].copy()
test = data[train_shape:].copy()

fe_col = [i for i in train.columns if i not in ['train_file_name', 'life', '开关2_sum', 'shebei',
'部件工作时长_skew',
 '累积量参数1_skew',
 '累积量参数2_skew',
 '转速信号1_skew',
 '转速信号2_skew',
 '压力信号1_skew',
 '压力信号2_skew',
 '温度信号_skew',
 '流量信号_skew',
 '电流信号_skew',
 '部件工作时长_diff_skew']]
X_train = train[fe_col].copy()
y_train = train['life'].copy()+10
X_test = test[fe_col].copy()

X_train_cp = X_train.copy()
y_train_cp = y_train.copy()
y_train_cp[np.where(y_train == 1638385)[0]] = 16383.85

y_train_cp = np.log1p(y_train_cp)

param = {
         'num_leaves': 512,
         'objective':'regression_l1',
         'max_depth': -1,
         'boosting': 'gbdt',
         'feature_fraction': 0.65,
         'bagging_fraction': 0.8,
         'bagging_freq':3,
         'metric': 'mse',
         'lambda_l2': 10,
         'nthread': -1,
         'verbosity': -1
}

if '开关1_switch' in X_train_cp.columns:
    del X_train_cp['开关1_switch'], X_test['开关1_switch']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
oof = np.zeros(len(X_train_cp))
predictions1 = np.zeros(len(X_test))
feature_importance_df = pd.DataFrame()
score = []
for i, (train_index, val_index) in enumerate(skf.split(X_train_cp,X_train_cp['设备'])):
    print("fold {}".format(i))
    X_tr, X_val = X_train_cp.iloc[train_index], X_train_cp.iloc[val_index]
    y_tr, y_val = y_train_cp.iloc[train_index], y_train_cp.iloc[val_index]
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val,y_val)
    num_round = 10000
    clf = lgb.train(param, lgb_train, num_round, valid_sets=[lgb_train, lgb_val],
                    verbose_eval=100, early_stopping_rounds=50,
                   )
    oof[val_index] = clf.predict(X_val, num_iteration=clf.best_iteration)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = clf.feature_name()
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions1 += clf.predict(X_test, num_iteration=clf.best_iteration) / skf.n_splits

print('train_score : ',mean_squared_error(y_train_cp, oof))

sub = test[['train_file_name']].copy()
sub['life'] = np.expm1(predictions1).round(2)
sub.to_csv('../data/submit/sub.csv',index=False)