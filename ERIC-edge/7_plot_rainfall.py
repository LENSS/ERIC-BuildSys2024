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

def plot(fn):
   
   date = fn.split('_')[1].split('.')[0]
   yr = int(date[:4])
   mon = int(date[5:7])
   d = int(date[8:10])
   
   print('\nPlotting', date)
   try:
      rainfall_test = pd.read_csv(input_path + fn, parse_dates = ['DateTime'])
   except:
      print("Error when read {}".format(input_path+fn))
      return
   
   # get cumulative rainfall
   # rain_true_total = []
   # linear_total = []
   # poly_total = []
   # dt_total = []
   # rf_total = []
   # ANN_total = []
   # svm_rbf_total = []

   # filter the ANN regress results based on the ANN rain detection results
   ANN_f = []
   for i in range(len(rainfall_test['ANN'])):
     
     if rainfall_test['ANN'][i] == 1: # based on ANN detection 
       ANN_f.append(rainfall_test['Regres_ANN'][i])
     else:
       ANN_f.append(0)
   
   rainfall_test['Regres_ANN_f'] = ANN_f
   
   model = 'Regres_ANN_f'
   label = 'ANN'
   
   t = rainfall_test['DateTime']
   y = rainfall_test['rainfall_true']
   y1 = rainfall_test[model]
   
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
   
   else:
      print('Unrecognized date {}'.format(date))
   
   axs.set_ylabel('Rain Intensity, mm/min')
   date_form = DateFormatter("%H:%M")
   axs.xaxis.set_major_formatter(date_form)
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
   
   rainfall_test['rainfall_true_total'] = y_total
   rainfall_test['Regres_ANN_f_total'] = y1_total
   
   fig, axs = plt.subplots(figsize=(8,6))
   
   axs.plot_date(datetime_raw, rain_cumu_raw, 'green',label='Rain Gauge', linestyle="None", marker='^', markersize=6, alpha = 1.0) # rain gauge cumulative
   axs.plot_date(t, y_total,'red',label="Interpolated", linestyle='solid',marker='None',markersize=3, alpha = 1) # True
   axs.plot_date(t, y1_total,'blue',label=label, linestyle='dashed',marker='None',markersize=3, alpha = 1) # pred
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
   

def split_date(fn):
      
   data = pd.read_csv(input_path + fn, parse_dates = ['DateTime'])
   
   # collect all possible date in the dataframe
   date_list = []
   for date in data['DateTime']:
      date_str = str(date)[:10]
      date_list.append(date_str)
   
   data['date_str'] = date_list
   
   date_set = sorted(set(date_list))
   print('unique dates found are:', date_set)
   
   # collect each date into a dataframe
   res = []
   for d in date_set:
      res.append(data.loc[data['date_str'] == d])
   
   # output each dataframe to a csv file
   for i, r in enumerate(res):
      fout = output_path + 'rainfall_' + date_set[i]+'.csv' 
      r.to_csv(fout, index=False)
      
   print("Split complete!")   
   
   
#------------------------------------------ main
with open('control.yaml') as f:
   controls = yaml.load(f, Loader=yaml.FullLoader)

input_path = controls['io_path']
output_path = controls['io_path']
fn_list = controls['rainfall_plot']

# load rain guage rain per day
with open('../process_rain/output/tamu_raindata_rainday_raw.json') as json_file:
   rain_day = json.load(json_file)

# load actual rain gauge record
with open('../process_rain/output/tamu_raindata_rainday_raw_cumul.json') as json_file:
   rain_day_cumul = json.load(json_file)

# split predictions by date
split_date('rainfall_val.csv')
split_date('rainfall_test.csv')

# plot day by day based on splitted data
for fn in fn_list:
   plot(fn)


