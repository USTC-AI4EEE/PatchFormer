# -*- coding: utf-8 -*-
import os
from assistant import get_gpus_memory_info
id,_ = get_gpus_memory_info()
os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from shutil import copyfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder,EncoderNormalizer,MultiNormalizer,TorchNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
import scipy.io
from sklearn.preprocessing import MinMaxScaler
# from pytorch_forecasting.data import GroupNormalizer
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('agg')
from datetime import datetime
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='PatchFormer', help='Model name.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--root_dir', type=str, default='NASA_RUL_prediction_sl_30', help='root path of the store file')
parser.add_argument('--seq_len', type=int, default=30, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')     
parser.add_argument('--patch_len', type=int, default=2, help='patch length for patch_embedding')     
parser.add_argument('--d_model', type=int, default=16, help='hidden dimensions of model')
parser.add_argument('--count', type=int, default=10, help='The number of independent experiment.')
parser.add_argument('--batch_size', type=int, default=16, help='The batch size.')
parser.add_argument('--data_dir', type=str, default='datasets/NASA/', help='path of the data file')
parser.add_argument('--Battery_list', type=list, default=['B0005', 'B0006', 'B0007', 'B0018'], help='Battery data')
parser.add_argument('--Rated_Capacity', type=float, default=2.0, help='Rate Capacity')
parser.add_argument('--test_name', type=str, default='B0005', help='Battery data used for test')
parser.add_argument('--start_point_list', type=int, default=[50,70,90], help='The cycle when prediction gets started.')
parser.add_argument('--max_epochs', type=int, default=200, help='max train epochs')
args = parser.parse_args()
# load .mat data
# convert str to datatime 
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split("/")[-1].split(".")[0]
    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0];
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(convert_to_time(col[i][2][0])), d2
        data.append(d1)

    return data
def getBatteryCapacityData(Battery,name):
    elem_list = []
    i = 1
    for Bat in Battery:
        elem = []
        if Bat['type'] == 'discharge':
            elem.append(name)
            elem.append(i)
            elem.append(Bat['data']['Capacity'][0])
            i += 1
            elem_list.append(elem)
    return elem_list
def DataRead(Battery_list,dir_path):
    BatteryData = []
    for name in Battery_list:
        print('Load Dataset ' + name + '.mat ...')
        path = dir_path + name + '.mat'
        data = loadMat(path)
        BatteryData += getBatteryCapacityData(data,name)
    return BatteryData
def DataProcess(BatteryData,test_name,start_point):
    df = pd.DataFrame(BatteryData, columns=['BatteryName', 'Cycle','Capacity'])
    df['Capacity'] /= args.Rated_Capacity
    # df.rename(columns = {'Capacity':'target'},inplace=True)
    df['constant'] = df['Capacity'] * 0   
    df['target'] = df['Capacity']
    df_test = df.loc[df['BatteryName']==test_name,['constant','Cycle','Capacity','target']]
    df_train = df.loc[(df['BatteryName']!=test_name) | ((df['BatteryName']==test_name) & (df['Cycle']<start_point)),['constant','Cycle','Capacity','target']]
    min_val = df_train['Capacity'].min()
    max_val = df_train['Capacity'].max()
    df_train['Capacity'] = (df_train['Capacity']-min_val) / (max_val - min_val)
    df_test['Capacity'] = (df_test['Capacity']-min_val) / (max_val - min_val)
    df_train['idx'] = [x for x in range(len(df_train))]
    df_train.set_index('idx',inplace=True)
    df_train['time_idx'] = df_train.index.to_series()
    df_test['idx'] = [x for x in range(len(df_test))]
    df_test.set_index('idx',inplace=True)
    df_test['time_idx'] = df_test.index.to_series()
    return df_train,df_test

#------------------------------------------------- step 1: 数据准备 ----------------------------------------
# -------------------------------数据分析和数据预处理【至关重要】---------------------------------------
BatteryData = DataRead(args.Battery_list,args.data_dir)
_,df_test = DataProcess(BatteryData,args.test_name,args.start_point_list[0])
real_data = df_test['target'].values*args.Rated_Capacity
if not os.path.exists('results'):
    os.makedirs('results')
torch.save(real_data, 'results/Capacity_{}_real_data.pth'.format(args.test_name))
