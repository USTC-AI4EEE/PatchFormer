import numpy as np
import pandas as pd
import argparse
import glob
import os
import torch
from assistant import get_gpus_memory_info
id,_ = get_gpus_memory_info()
os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
def drop_outlier(array,count,bins):
    index = []
    range_ = np.arange(1,count,bins)
    for i in range_[:-1]:
        array_lim = array[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max,th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)
def BatteryDataRead(Battery_list,dir_path):
    Battery = {}
    for name in Battery_list:
        print('Load Dataset ' + name + ' ...')
        path = glob.glob(dir_path + name + '/*.xlsx')
        dates = []
        for p in path:
            df = pd.read_excel(p, sheet_name=1)
            print('Load ' + str(p) + ' ...')
            dates.append(df['Date_Time'][0])
        idx = np.argsort(dates)
        path_sorted = np.array(path)[idx]
        
        count = 0
        discharge_capacities = []
        health_indicator = []
        internal_resistance = []
        CCCT = []
        CVCT = []
        for p in path_sorted:
            df = pd.read_excel(p,sheet_name=1)
            print('Load ' + str(p) + ' ...')
            cycles = list(set(df['Cycle_Index']))
            for c in cycles:
                df_lim = df[df['Cycle_Index'] == c]
                #Charging
                df_c = df_lim[(df_lim['Step_Index'] == 2)|(df_lim['Step_Index'] == 4)]
                c_v = df_c['Voltage(V)']
                c_c = df_c['Current(A)']
                c_t = df_c['Test_Time(s)']
                #CC or CV
                df_cc = df_lim[df_lim['Step_Index'] == 2]
                df_cv = df_lim[df_lim['Step_Index'] == 4]
                CCCT.append(np.max(df_cc['Test_Time(s)'])-np.min(df_cc['Test_Time(s)']))
                CVCT.append(np.max(df_cv['Test_Time(s)'])-np.min(df_cv['Test_Time(s)']))

                #Discharging
                df_d = df_lim[df_lim['Step_Index'] == 7]
                d_v = df_d['Voltage(V)']
                d_c = df_d['Current(A)']
                d_t = df_d['Test_Time(s)']
                d_im = df_d['Internal_Resistance(Ohm)']

                if(len(list(d_c)) != 0):
                    time_diff = np.diff(list(d_t))
                    d_c = np.array(list(d_c))[1:]
                    discharge_capacity = time_diff*d_c/3600 # Q = A*h
                    discharge_capacity = [np.sum(discharge_capacity[:n]) for n in range(discharge_capacity.shape[0])]
                    discharge_capacities.append(-1*discharge_capacity[-1])

                    dec = np.abs(np.array(d_v) - 3.8)[1:]
                    start = np.array(discharge_capacity)[np.argmin(dec)]
                    dec = np.abs(np.array(d_v) - 3.4)[1:]
                    end = np.array(discharge_capacity)[np.argmin(dec)]
                    health_indicator.append(-1 * (end - start))

                    internal_resistance.append(np.mean(np.array(d_im)))
                    count += 1

        discharge_capacities = np.array(discharge_capacities)
        health_indicator = np.array(health_indicator)
        internal_resistance = np.array(internal_resistance)
        CCCT = np.array(CCCT)
        CVCT = np.array(CVCT)
        
        idx = drop_outlier(discharge_capacities, count, 40)    
        df_result = pd.DataFrame({'BatteryName':[name for i in range(idx.shape[0])],
                                'Cycle':np.linspace(1,idx.shape[0],idx.shape[0]),
                                'Capacity':discharge_capacities[idx],
                                'SoH':health_indicator[idx],
                                'Resistance':internal_resistance[idx],
                                'CCCT':CCCT[idx],
                                'CVCT':CVCT[idx]})
        Battery[name] = df_result
    return Battery

def BatteryDataProcess(BatteryData,test_name,start_point,args):
    dict_without_test = {key: value for key, value in BatteryData.items() if key != test_name}
    df_train = pd.concat(dict_without_test.values())
    df_train = df_train.filter(items=['BatteryName', 'Cycle','Capacity'])
    df_train['Capacity'] /= args.Rated_Capacity
    df_train['target'] = df_train['Capacity']
    df_train['time_idx'] = df_train['Cycle'].map(lambda x: int(x-1))
    df_train['group_id'] = df_train['BatteryName'].map({'CS2_35':0, 'CS2_36':1, 'CS2_37':2, 'CS2_38':3})
    df_train = df_train.drop(['BatteryName'],axis=1)
    df_train['idx'] = [x for x in range(len(df_train))]
    df_train.set_index('idx',inplace=True)

    df_test = BatteryData[test_name]
    df_test = df_test.filter(items=['BatteryName', 'Cycle','Capacity'])
    df_test['Capacity'] /= args.Rated_Capacity
    df_test['target'] = df_test['Capacity']
    df_test['time_idx'] = df_test['Cycle'].map(lambda x: int(x-1))
    df_test['group_id'] = df_test['BatteryName'].map({'CS2_35':0, 'CS2_36':1, 'CS2_37':2, 'CS2_38':3})
    df_test = df_test.drop(['BatteryName'],axis=1)
    df_test['idx'] = [x for x in range(len(df_test))]
    df_test.set_index('idx',inplace=True)

    min_val = df_train['Capacity'].min()
    max_val = df_train['Capacity'].max()
    df_train['Capacity'] = (df_train['Capacity']-min_val) / (max_val - min_val)
    df_test['Capacity'] = (df_test['Capacity']-min_val) / (max_val - min_val)
    #测试集全部数据
    df_all = df_test 
    #测试集参与测试的数据
    df_test = df_all.loc[df_all['Cycle']>=start_point-args.seq_len,['time_idx','group_id','Cycle','Capacity','target']] 
    df_test['idx'] = [x for x in range(len(df_test))]
    df_test.set_index('idx',inplace=True)
    return df_train,df_test,df_all



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CALCE', help='the name of dataset')
    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length') 
    parser.add_argument('--Battery_list', type=list, default=['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38'], help='Battery data')
    parser.add_argument('--data_dir', type=str, default='data/CALCE data/', help='path of the data file')
    parser.add_argument('--Rated_Capacity', type=float, default=1.1, help='Rate Capacity')
    parser.add_argument('--test_name', type=str, default='CS2_35', help='Battery data used for test')
    parser.add_argument('--start_point_list', type=int, default=[300,400,500], help='The cycle when prediction gets started.')
    args = parser.parse_args()

    BatteryData = BatteryDataRead(args.Battery_list,args.data_dir)
    BatteryData_array = np.array([BatteryData], dtype=object)
    np.save('data/CALCE data/CALCE_Data.npy', BatteryData_array, allow_pickle=True)
    BatteryData = np.load('data/CALCE data/CALCE_Data.npy', allow_pickle=True)
    BatteryData = BatteryData.item()
    _,_,df_all = BatteryDataProcess(BatteryData,args.test_name,args.start_point_list[0],args)
    real_data = df_all['target'].values*args.Rated_Capacity
    if not os.path.exists('results'):
        os.makedirs('results')
    torch.save(real_data, 'results/Capacity_{}_real_data.pth'.format(args.test_name))