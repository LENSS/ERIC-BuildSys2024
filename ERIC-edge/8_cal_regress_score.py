#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 00:48:23 2022

@author: tian
"""

# calculate the regression score based on filtered estimation with results from rain detection

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from matplotlib.dates import DateFormatter
import yaml
import json
import sklearn.metrics as sm

def filter_by_detection(data):
   
   cols = ['Regres_Linear','Regres_Poly','Regres_RF','Regres_SVR-rbf','Regres_ANN']
   #cols = ['Regres_RF','Regres_ANN'] # for easy debug use only

   result = dict()
   f_cols = [] 
   for c in cols:
      f_c = c+'_f'
      # print(f_c)
      f_cols.append(f_c)
      f_data = []
      if c == 'Regres_RF':
         key = 'RandomForest'
      else:
         key = 'ANN'
      print("column_name:", c)
      print("key:", key)

      for i, entry in enumerate(data[c]):
         if data[key][i] == 1: # all use ANN detection results, here use RF detection for RF regressor
            f_data.append(entry)
         else:
            f_data.append(0)
      # print('len(f_data) =', len(f_data))
      
      result[f_c] = f_data
   
   #build dataframe
   result = pd.DataFrame.from_dict(result)
   result['ANN'] = data['ANN'] # these are two filters
   result['RandomForest'] = data['RandomForest'] # these are two filters
   result['DateTime'] = data['DateTime']
   result['rainfall_true'] = data['rainfall_true']
   
   return result, f_cols

def split_date(data, method):
     
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
    

def cal_avg_daily_diff(data_f, method):
   
   data = data_f[['DateTime', 'rainfall_true', method]].copy()
   
   # split by date
   daily_rain = split_date(data, method)
   
   diff_list = []
   for k, v in daily_rain.items():
      diff = abs(sum(v[method]) - sum(v['rainfall_true'])) # calculate the err for each date
      print(k, diff)
      diff_list.append(diff)
   
   avg_diff = sum(diff_list)/len(diff_list) # calculate the average err
      
   return avg_diff

def cal_score(data_f, method, fw):
   
    y_test = data_f['rainfall_true']
    y_pred = data_f[method]
    
    s1= sum(y_test)
    s2= sum(y_pred)
    error_pct = abs(s1-s2)/s1
    
    avg_diff = cal_avg_daily_diff(data_f, method)
    
    MAE = round(sm.mean_absolute_error(y_test, y_pred), 4)
    MSE = round(sm.mean_squared_error(y_test, y_pred), 4)
    MDAE = round(sm.median_absolute_error(y_test, y_pred), 4)
    EVS = round(sm.explained_variance_score(y_test, y_pred), 4)
    R2_Score = round(sm.r2_score(y_test, y_pred), 4)
       
    fw.write('%16s' % method); fw.write("\t")
    fw.write('%16.4f' % s1); fw.write("\t")
    fw.write('%16.4f' % s2); fw.write("\t")
    fw.write('%16.4f' % error_pct); fw.write("\t")
    fw.write('%16.4f' % avg_diff); fw.write("\t")
    fw.write('%16.4f' % MAE); fw.write("\t")
    fw.write('%16.4f' % MSE); fw.write("\t")
    fw.write('%16.4f' % MDAE); fw.write("\t")
    fw.write('%16.4f' % EVS); fw.write("\t")
    fw.write('%16.4f' % R2_Score)
    fw.write("\n")

    print(method)
    print("sum of y_test = ", s1)
    print("sum of y_pred = ", s2)
    print("Total_Rel_Err = ", error_pct)
    print("Avg_Daily_Err =", avg_diff)
    print("Mean absolute error =", MAE)
    print("Mean squared error =", MSE)
    print("Median absolute error =", MDAE)
    print("Explain variance score =", EVS)
    print("R2-Score =", R2_Score)
    print('--'*20)

def cal_regress_score(data_f, cols_f, mode):
   
   if mode == 'val':
      fw = open(output_path+'val_regress_score_filtered.csv', 'w')
   elif mode == 'test':
      fw = open(output_path+'test_regress_score_filtered.csv', 'w')
   else:
      print("unrecognized mode")
      raise ValueError
   
   # write table head
   col_names = ['Model','sum_y_val','sum_y_pred_f','Total_Rel_Err', 'Avg_Daily_Err', 'MAE_f', 'MSE_f', 'MDAE_f','EVS_f','R2_Score_f']
   for name in col_names:
      fw.write('%16s' % name); fw.write("\t")
   fw.write('\n')
   
   # calculate the scores based on filtered data
   for c in cols_f:
      cal_score(data_f, c, fw)
      
   fw.close()
   
#------------------------------------------ main
input_path = '../../2021_RaduHome_VideoData/RS_ML/output/'
output_path = input_path

fn_val = input_path + 'rainfall_val_visual.csv'
fn_test = input_path + 'rainfall_test_visual.csv'

val = pd.read_csv(fn_val, parse_dates = ['DateTime'])
test = pd.read_csv(fn_test, parse_dates = ['DateTime'])

# filter rainfall estimation result by ANN detection result
val_f, val_cols = filter_by_detection(val)

val_f.to_csv(input_path+'rainfall_val_visual_filtered.csv', index=False)

test_f, test_cols = filter_by_detection(test)
test_f.to_csv(input_path+'rainfall_test_visual_filtered.csv', index=False)


cal_regress_score(val_f, val_cols, 'val')
print()
print()
cal_regress_score(test_f, test_cols, 'test')