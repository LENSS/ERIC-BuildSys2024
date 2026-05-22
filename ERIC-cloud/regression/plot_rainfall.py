#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 18:48:14 2021

@author: tian
"""

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from matplotlib.dates import DateFormatter
import yaml
import json
from plot_israin import parse_datetime
import sklearn.metrics as sm


def plot(fn):
   
   date = fn.split('_')[1].split('.')[0]
   yr = int(date[:4])
   mon = int(date[5:7])
   d = int(date[8:10])
   
   print('\nPlot', date)
   try:
      rainfall_test = pd.read_csv(input_path + fn, parse_dates = ['DateTime'])
   except:
      print("Error when read {}".format(input_path+fn))
      return
   
   # filter the regress results based on the rain detection results, on a single img basis
   rainfall_f = []
   for i in range(len(rainfall_test['label'])):
     if rainfall_test['predicted'][i] == 1: # based on rain detection 
       rainfall_f.append(max(rainfall_test['predicted_regress'][i],0)) # fix the negative prediction
     else:
       rainfall_f.append(0)
   
   rainfall_test['predicted_regress_f'] = rainfall_f
   
   # reorder the columns of rainfall_test for grouping
   new_cols = \
      ['img_name','DateTime','date_str','predicted','label','predicted_regress','label_regress','predicted_regress_f']
   rainfall_test = rainfall_test[new_cols]
    
   # average by minute
   # rainfall_test = rainfall_test.groupby(pd.Grouper(key='DateTime',freq='1.0min'))[rainfall_test.columns[3:]].mean()
   # rainfall_test = rainfall_test.groupby(pd.Grouper(key='DateTime',freq='1.0min'))[rainfall_test.columns[3:]].max()
   
   # for 5s labels, take summation
   # rainfall_test = rainfall_test.groupby(pd.Grouper(key='DateTime',freq='1.0min'))[rainfall_test.columns[3:]].sum()
   
   # take mean for the minute
   rainfall_test = rainfall_test.groupby(pd.Grouper(key='DateTime',freq='1.0min'))[rainfall_test.columns[3:]].mean() # +++++ average by minute, avg of 12 img

   rainfall_test.reset_index(level=0, inplace=True) # reset the timelabel from index to column
   
   rainfall_test.dropna(axis=0, inplace=True, how='any') # drop the rows with NAN values
   rainfall_test.reset_index(drop=True, inplace=True)   
   
   # this is for 5s labels
   # rainfall_pred_f_avg_12 = []
   # label_12 = []
   
   # for i, val in enumerate(rainfall_test['predicted_regress_f']):
   #    rainfall_pred_f_avg_12.append(val*12)
   #    label_12.append(rainfall_test['label_regress'][i]*12)
   
   # rainfall_test['predicted_regress_f'] = rainfall_pred_f_avg_12
   # rainfall_test['label_regress'] = label_12
   
   # save the new dataset for score calculation
   rainfall_test.to_csv(output_path+fn.split('.')[0]+'_f.csv', index=False) # +++++++++++++++++ save the filtered rainfall prediction

   label = 'ResNet18'
   
   t = rainfall_test['DateTime']
   y = rainfall_test['label_regress']
   
   y1 = rainfall_test['predicted_regress_f']
   
   #------------------ plot rain intensity vs. time
   fig, axs = plt.subplots(figsize=(8,6))
   axs.plot_date(t, y,'red',label="Interpolated", linestyle='solid',marker='o',markersize=3, alpha = 1)
   axs.plot_date(t, y1,'blue',label=label, linestyle='dashed',marker='o',markersize=3, alpha = 0.5)
   
   axs.set_xlabel(date)
   
   if date == '2021-08-15':
      #20210815
      axs.set_xlim([dt.datetime(year=yr,month=mon,day=d,hour=13,minute=0),
                    dt.datetime(year=yr,month=mon,day=d,hour=16,minute=30)])
   
   elif date == '2021-08-18':
      # 2021-08-18
      axs.set_xlim([dt.datetime(year=yr,month=mon,day=d,hour=10,minute=30),
                    dt.datetime(year=yr,month=mon,day=d,hour=18,minute=0)])

      # axs.set_xlim([dt.datetime(year=yr,month=mon,day=d,hour=10,minute=30),
      #               dt.datetime(year=yr,month=mon,day=d,hour=12,minute=0)])
      # axs.set_xlim([dt.datetime(year=yr,month=mon,day=d,hour=14,minute=0),
      #               dt.datetime(year=yr,month=mon,day=d,hour=18,minute=30)])
      
   # elif date == '2021-07-06':
   #    axs.set_xlim([dt.datetime(year=yr,month=mon,day=d,hour=8,minute=0),
   #                  dt.datetime(year=yr,month=mon,day=d,hour=17,minute=0)])
   
   # elif date == '2021-07-12':
   #    axs.set_xlim([dt.datetime(year=yr,month=mon,day=d,hour=11,minute=0),
   #                  dt.datetime(year=yr,month=mon,day=d,hour=14,minute=0)])
   
   # else:
   #    print('Unrecognized date {}'.format(date))
   
   axs.set_ylabel('Rain Intensity, mm/min')
   date_form = DateFormatter("%H:%M")
   axs.xaxis.set_major_formatter(date_form)
   # plt.ylim((0,3.0))
   plt.legend()
   plt.savefig(output_path+date+'_rain-int.png')
   # plt.show()
   
   #------------------------------------- plot cumulative rainfall amount vs. time
   # get raw rain cumulative records
   rain_day_dict = rain_day_cumul[date]
   time_raw = [k for k,_ in rain_day_dict.items()]
   rain_cumu_raw = [v for _,v in rain_day_dict.items()]
   
   datetime_raw = [dt.datetime.strptime(t,"%Y-%m-%d %H:%M:%S") for t in time_raw]
      
   y_total = []
   y_total.append(y[0])
   
   y1_total = []
   y1_total.append(y1[0])
   
   for i in range(1,len(y1)):
     y_total.append(y_total[-1]+y[i])
     y1_total.append(y1_total[-1]+y1[i])
   
   rainfall_test['label_total'] = y_total
   rainfall_test['predicted_regress_f_total'] = y1_total
   
   fig, axs = plt.subplots(figsize=(8,6))
   
   axs.plot_date(datetime_raw, rain_cumu_raw, 'green',label='Rain Gauge', linestyle="None", marker='^', markersize=6, alpha = 1.0)
   axs.plot_date(t, y_total,'red',label="Interpolated", linestyle='solid',marker='None',markersize=3, alpha = 1)
   axs.plot_date(t, y1_total,'blue',label=label, linestyle='dashed',marker='None',markersize=3, alpha = 1)
   
   axs.set_xlabel(date)
   
   if date == '2021-08-15':
      #20210815
      axs.set_xlim([dt.datetime(year=yr,month=mon,day=d,hour=13,minute=0),
                    dt.datetime(year=yr,month=mon,day=d,hour=16,minute=30)])
   
   elif date == '2021-08-18':
      # 2021-08-18
      axs.set_xlim([dt.datetime(year=yr,month=mon,day=d,hour=10,minute=30),
                    dt.datetime(year=yr,month=mon,day=d,hour=18,minute=0)])
      
   # elif date == '2021-07-06':
   #    axs.set_xlim([dt.datetime(year=yr,month=mon,day=d,hour=8,minute=0),
   #                  dt.datetime(year=yr,month=mon,day=d,hour=17,minute=0)])
      
   # elif date == '2021-07-12':
   #    axs.set_xlim([dt.datetime(year=yr,month=mon,day=d,hour=11,minute=0),
   #                  dt.datetime(year=yr,month=mon,day=d,hour=14,minute=0)])
      
   # else:
   #    print('Unrecognized date {}'.format(date))
      
   # axs.set_ylim([0,60])
   axs.set_ylabel('Cumulative Rainfall, mm')
   date_form = DateFormatter("%H:%M")
   axs.xaxis.set_major_formatter(date_form)
   plt.ylim((0,80))
   plt.legend()
   
   # save the cumulative rain results
   # rainfall_test.to_csv(output_path + 'rainfall_ANN.csv', index=False)
   
   raw = rain_day[date]
   interpolated = y_total[-1]
   predicted = y1_total[-1]
   abs_err = abs(y_total[-1]-y1_total[-1])
   rel_err = abs(y_total[-1]-y1_total[-1])/y_total[-1]
   
   print("Rain_gauge = {:.2f} mm ({:.2f} inches)".format(raw, raw/25.4))
   print("Interpolated cul_rain = {:.2f} mm ({:.2f} inches)".format(interpolated, interpolated/25.4 ))
   print("Predicted cul_rain = {:.2f} mm ({:.2f} inches)".format(predicted, predicted/25.4))
   print("Abs_diff = {:.2f} mm ({:.2f} inches)".format(abs_err, abs_err/25.4))
   print("Relative Error = {:.4f}".format(rel_err))
   
   textstr = "Rain Gauge = {:.2f} mm\nInterpolated = {:.2f} mm\nPredicted = {:.2f} mm\nAbs_diff = {:.2f} mm".format \
            (raw, interpolated, predicted, abs_err)
            
   props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
   
   axs.text(0.6, 0.2, textstr, transform=axs.transAxes, fontsize=11,
           verticalalignment='top', bbox=props)
   plt.savefig(output_path+date+'_rain-cul.png')
   # plt.show()   

def split_date(fn, fn_raindetect, mode):
      
   data = pd.read_csv(input_path + fn)
   
   rain_detect = pd.read_csv(fn_raindetect)

   # remove the ["] wrapped outside the img_name
   clean_img_name = []
   for name in data['img_name']:
      name = name.strip("[]'")[:]
      clean_img_name.append(name)
   
   data['img_name'] = clean_img_name
   
   print('data.head:')
   print(data.head())
   print('rain_detect.head:')
   print(rain_detect.head())

   # merge with rain detection result
   print("before merge, len(regress) = {}, len(class) = {}".format(data.shape[0], rain_detect.shape[0]))
   
   data = pd.merge(data, rain_detect, on='img_name', how='inner')
   print("after merge, len(data) = {}".format(data.shape[0]))
   
   # filter the estimated rainfall intensity by rain detection results, on a single img basis
   rainfall_regress_f = []
   for i in range(len(data['predicted_regress'])):
      if data['predicted'][i]==1:
         rainfall_regress_f.append(max(data['predicted_regress'][i],0)) # avoid negative rainfall
      else:
         rainfall_regress_f.append(0)
   
   # cal_score(data['label_regress'], rainfall_regress_f, mode)
   
   
   # collect all possible date in the dataframe
   date_list = []
   date_time_list = []
   for date in data['img_name']:
      # 2021-8-15-10-0-1-15-1-rgb.png
      date_time = parse_datetime(date)      
      date_time_list.append(date_time)
      date_str = date_time.strftime("%Y-%m-%d")
      date_list.append(date_str)
   
   data['date_str'] = date_list
   data['DateTime'] = date_time_list
   
   date_set = sorted(set(date_list))
   print('unique dates found are:', date_set)
   
   # collect each date into a dataframe
   res = []
   for d in date_set:
      single_day = data.loc[data['date_str'] == d]
      
      # # average by minute
      # single_day_avg = single_day.groupby(pd.Grouper(key='DateTime',freq='1.0min'))[single_day.columns[:-1]].mean()
      # single_day_avg.reset_index(level=0, inplace=True) # reset the timelabel from index to column
      # single_day_avg.dropna(axis=0, inplace=True, how='any') # drop the rows with NAN values
      # single_day_avg.reset_index(drop=True, inplace=True)
  
      res.append(single_day)
   
   # output each dataframe to a csv file
   for i, r in enumerate(res):
      fout = output_path + 'rainfall_' + date_set[i]+'.csv' 
      r.to_csv(fout, index=False)
      
   print("Split complete!")   
   
#------------------------------------------ main
with open('../control.yaml') as f:
   controls = yaml.load(f, Loader=yaml.FullLoader)

input_path = controls['io_path']
output_path = controls['io_path']
fn_list = controls['rainfall_plot']
rain_detect_folder = controls['rain_detect_folder']

# load rain guage rain per day
with open('../labels/tamu_raindata_rainday_raw.json') as json_file:
   rain_day = json.load(json_file)

# load actual rain gauge record
with open('../labels/tamu_raindata_rainday_raw_cumul.json') as json_file:
   rain_day_cumul = json.load(json_file)

split_date('regress_val_res.csv', rain_detect_folder+'val_pred_res.csv', 'val')
split_date('regress_test_res.csv', rain_detect_folder+'test_pred_res.csv','test')

for fn in fn_list:
   plot(fn)

