#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 00:02:54 2022

@author: tian
"""
import pandas as pd
import yaml
import sklearn.metrics as sm


def split_date(data):
     
   # collect all possible date in the dataframe
   date_list = []
   for date in data['DateTime']:
      date_str = str(date)[:10]
      date_list.append(date_str)
   
   data['date_str'] = date_list
   
   date_set = sorted(set(date_list))
   print('unique dates found are:', date_set)
   
   # collect each date into a dataframe
   daily_rain = dict()
   for d in date_set:
      daily_rain[d] = data.loc[data['date_str'] == d]
   
   return daily_rain

def cal_avg_daily_diff(data_f):
   
   data = data_f[['DateTime', 'label_regress', 'predicted_regress_f']].copy()
   
   # split by date
   daily_rain = split_date(data)
   
   diff_list = []
   for k, v in daily_rain.items():
      diff = abs(sum(v['predicted_regress_f']) - sum(v['label_regress']))
      print(k, diff)
      diff_list.append(diff)
   
   avg_diff = sum(diff_list)/len(diff_list)
      
   return avg_diff


def cal_score(data_f, y_test, y_pred, mode):
   
    print(mode)
    s1= sum(y_test)
    s2= sum(y_pred)
    error_pct = abs(s1-s2)/s1
    avg_diff = cal_avg_daily_diff(data_f)
    
    MAE = round(sm.mean_absolute_error(y_test, y_pred), 4)
    MSE = round(sm.mean_squared_error(y_test, y_pred), 4)
    MDAE = round(sm.median_absolute_error(y_test, y_pred), 4)
    EVS = round(sm.explained_variance_score(y_test, y_pred), 4)
    R2_Score = round(sm.r2_score(y_test, y_pred), 4)

    if mode == 'val':
       fw = fw_val
    elif mode == 'test':
       fw = fw_test
       
    fw.write('%15s' % method); fw.write("\t")
    fw.write('%15.4f' % s1); fw.write("\t")
    fw.write('%15.4f' % s2); fw.write("\t")
    fw.write('%15.4f' % error_pct); fw.write("\t")
    fw.write('%15.4f' % avg_diff); fw.write("\t")
    fw.write('%15.4f' % MAE); fw.write("\t")
    fw.write('%15.4f' % MSE); fw.write("\t")
    fw.write('%15.4f' % MDAE); fw.write("\t")
    fw.write('%15.4f' % EVS); fw.write("\t")
    fw.write('%15.4f' % R2_Score)
    fw.write("\n")

    print("sum of y_test = ", s1)
    print("sum of y_pred = ", s2)
    print("error pct = ", error_pct)
    print("avg_daily_diff =", avg_diff)
    print("Mean absolute error =", MAE)
    print("Mean squared error =", MSE)
    print("Median absolute error =", MDAE)
    print("Explain variance score =", EVS)
    print("R2 score =", R2_Score)
    print('--'*20)

def combine_data(files):
   
   data_list = []
   for file in files:
      data = pd.read_csv(input_path+file, parse_dates=['DateTime'])
      data_list.append(data)
   
   # merge all datasets
   df = pd.concat(data_list, ignore_index=True)

   # sort by the time
   df = df.sort_values(by=['DateTime'], ascending = True)

   return df
  

#---------------------------main()

val_files = [\
'rainfall_2021-08-15_f.csv',
'rainfall_2021-08-18_f.csv',
'rainfall_2021-09-03_f.csv',
'rainfall_2021-09-13_f.csv'
]
   
test_files = [\
'rainfall_2021-09-18_f.csv',
'rainfall_2021-09-28_f.csv',
'rainfall_2021-09-29_f.csv',
'rainfall_2021-10-01_f.csv',
]

with open('../control.yaml') as f:
   controls = yaml.load(f, Loader=yaml.FullLoader)

input_path = controls['io_path']
output_path = controls['io_path']
method = controls['model']

# log file
col_names = ['Model','sum_y_val','sum_y_pred_f','error_pct','avg_daily_diff_f','MAE', 'MSE', 'MDAE','EVS','R2_Score']

fw_val = open(output_path + method + '_val_rain_estimation_score_filtered.log','w')
for name in col_names:
    fw_val.write('%15s' % name); fw_val.write("\t")
fw_val.write('\n')

fw_test = open(output_path + method + '_test_rain_estimation_score_filtered.log','w')
for name in col_names:
    fw_test.write('%15s' % name); fw_test.write("\t")
fw_test.write('\n')

val_data = combine_data(val_files)
test_data = combine_data(test_files)

cal_score(val_data, val_data['label_regress'], val_data['predicted_regress_f'], 'val')
cal_score(test_data, test_data['label_regress'], test_data['predicted_regress_f'], 'test')

fw_val.close()
fw_test.close()
