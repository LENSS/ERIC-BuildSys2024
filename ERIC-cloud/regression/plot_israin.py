#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 16:11:36 2022

@author: tian
"""

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from matplotlib.dates import DateFormatter

import yaml


def parse_datetime(img_name):
   '../tamu_videos/val_rgb/norain/2021-8-15-10-0-1-15-1-rgb.png'
   
   time_str = img_name.split('/')[-1].split('.')[0].split('-')[:6]
   time_str = '-'.join(time_str)
   res = dt.datetime.strptime(time_str, '%Y-%m-%d-%H-%M-%S')
   
   return res

def main():

   with open('../control.yaml') as f:
      controls = yaml.load(f, Loader=yaml.FullLoader)
   
   input_path = controls['io_path']
   
   fn1 = 'val_pred_res.csv'
   fn2 = 'test_pred_res.csv'
   result1 = pd.read_csv(input_path + fn1)
   result2 = pd.read_csv(input_path + fn2)
   result = pd.concat([result1, result2], ignore_index=True)
   
   result['DateTime'] = [parse_datetime(i) for i in result['img_name']]
   result = result.sort_values(by = 'DateTime', ascending = True)
   
   rain_raw = pd.read_csv('../labels/tamu_raindata_raw.csv', parse_dates=['Date'])
   rain_raw['israin'] = [1 for i in range(rain_raw.shape[0])]
   
   t = result['DateTime']
   y = result['label']
   model = 'ResNet18'
   y1 = result['predicted']
   
   for fn in controls['rainfall_plot']:
      date = fn.split('_')[1].split('.')[0]
      
      fig, axs = plt.subplots(figsize=(8,6))
      
      axs.plot_date(rain_raw['Date'], rain_raw['israin'],'green',label="Rain Gauge", linestyle='None',marker='^',markersize=3, alpha = 1)
      axs.plot_date(t, y*0.98,'red',label="Interpolated", linestyle='solid',marker='o',markersize=3, alpha = 1)
      axs.plot_date(t, y1*0.96,'blue',label=model, linestyle='dotted',marker='o',markersize=3, alpha = 1)
      
      
      axs.set_xlabel(date)
      if date == '2021-08-15':
         axs.set_xlim([dt.datetime(year=2021,month=8,day=15,hour=13,minute=0),
                       dt.datetime(year=2021,month=8,day=15,hour=16,minute=30)])
      
      elif date == '2021-08-18':
         axs.set_xlim([dt.datetime(year=2021,month=8,day=18,hour=10,minute=30),
                       dt.datetime(year=2021,month=8,day=18,hour=18,minute=0)])
      
      elif date == '2021-09-03':
         axs.set_xlim([dt.datetime(year=2021,month=9,day=3,hour=14,minute=00),
                       dt.datetime(year=2021,month=9,day=3,hour=16,minute=0)])
   
      elif date == '2021-09-13':
         axs.set_xlim([dt.datetime(year=2021,month=9,day=13,hour=16,minute=00),
                       dt.datetime(year=2021,month=9,day=13,hour=20,minute=0)])
   
      elif date == '2021-09-18':
         axs.set_xlim([dt.datetime(year=2021,month=9,day=18,hour=13,minute=30),
                       dt.datetime(year=2021,month=9,day=18,hour=15,minute=0)])
   
      elif date == '2021-09-28':
         axs.set_xlim([dt.datetime(year=2021,month=9,day=28,hour=8,minute=00),
                       dt.datetime(year=2021,month=9,day=28,hour=23,minute=59)])
   
      elif date == '2021-09-29':
         axs.set_xlim([dt.datetime(year=2021,month=9,day=29,hour=0,minute=00),
                       dt.datetime(year=2021,month=9,day=29,hour=8,minute=0)])
   
      elif date == '2021-10-01':
         axs.set_xlim([dt.datetime(year=2021,month=10,day=1,hour=4,minute=00),
                       dt.datetime(year=2021,month=10,day=1,hour=16,minute=0)])
   
      axs.set_ylabel('is_rain')
      
      date_form = DateFormatter("%H:%M")
   
      axs.xaxis.set_major_formatter(date_form)
      
      plt.legend()
      plt.savefig(input_path+date+'_is_rain.png')
      
      plt.show()

if __name__ == '__main__':
   main()