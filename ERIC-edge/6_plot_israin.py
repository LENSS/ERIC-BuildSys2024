#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:02:49 2021

@author: tian
"""

"""
plot the testing result predication vs. true
"""

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from matplotlib.dates import DateFormatter

import yaml

with open('control.yaml') as f:
   controls = yaml.load(f, Loader=yaml.FullLoader)

input_path = controls['io_path']

fn = 'israin_val.csv'

result = pd.read_csv(input_path + fn, parse_dates = ['DateTime'])

t = result['DateTime']
y = result['is_rain_true']
model = 'ANN'
y1 = result[model]

fig, axs = plt.subplots()
axs.plot_date(t, y,'red',label="True", linestyle='solid',marker='o',markersize=3, alpha = 1)
axs.plot_date(t, y1,'blue',label=model, linestyle='dotted',marker='o',markersize=3, alpha = 0.3)


# axs.set_xlabel('2021-08-15')
# axs.set_xlim([dt.datetime(year=2021,month=8,day=15,hour=13,minute=0),
#               dt.datetime(year=2021,month=8,day=15,hour=16,minute=30)])

# axs.set_xlabel('2021-08-18')
# axs.set_xlim([dt.datetime(year=2021,month=8,day=18,hour=10,minute=30),
#               dt.datetime(year=2021,month=8,day=18,hour=11,minute=30)])

axs.set_xlabel('2021-08-18')
axs.set_xlim([dt.datetime(year=2021,month=8,day=18,hour=10,minute=30),
              dt.datetime(year=2021,month=8,day=18,hour=18,minute=0)])

# axs.set_xlabel('2021-08-18')
# axs.set_xlim([dt.datetime(year=2021,month=8,day=18,hour=16,minute=0),
#               dt.datetime(year=2021,month=8,day=18,hour=18,minute=0)])

axs.set_ylabel('is_rain')

date_form = DateFormatter("%H:%M")
# date_form = DateFormatter("%D")
axs.xaxis.set_major_formatter(date_form)

plt.legend()

plt.show()
