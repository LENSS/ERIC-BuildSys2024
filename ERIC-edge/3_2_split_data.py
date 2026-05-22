#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 22:13:21 2021

@author: tian
"""

# split data into train eval test

import pandas as pd
import math
import yaml
import os

with open('control.yaml') as f:
   controls = yaml.load(f, Loader=yaml.FullLoader)

input_path = controls['io_path']
output_path = controls['io_path']
fea_path = controls['fea_path']

fn = controls['fea_fn']
fea_suffix = controls['fea_suffix']
time_suffix = controls['time_suffix']

if not os.path.isdir(output_path):
  os.mkdir(output_path)
  print("Folder {} created".format(output_path))
else:
  print("Folder {} already existed".format(output_path))
  
# read in data
dataset = pd.read_csv(fea_path + fn)

#----------------------- split the dataset into train, eval and test set

# X = data.values
# y = dataset.loc[:,'is_rain'].values

# random split
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(X[:30409,:], y[:30409], test_size=0.1, random_state=0, stratify=y[:30409])
#x_train, x_test, y_train, y_test = train_test_split(X[:30409,:], y[:30409], test_size=0.1)

# sequential split
if time_suffix == '_day':
  split_point = 13736 # May Jun Jul for train, Aug for test
  end_point = 15206 # remove Sep

elif time_suffix == '_night':
  split_point = 12515 # May for train, June July Aug for test
  end_point = 19475 # remove Sep

elif time_suffix == '_all':
  
  # 30s interval
  #split_point = 45324 # May Jun Jul for train, Aug for test
  #end_point = 49884 # remove Sep
  
  # 60s interval
  # May Jun Jul for train, Aug for test
   # split_point = 22665
   # end_point =  24645
  
  # May Jun for train, Jul Aug for test
  # split_point = 17955 # End of June
  # split_point = 19906 # upto Jul 04
  # split_point = 20446 # upto Jul 05
  # split_point = 13966 # upto end of may
  # end_point = 24645
  
  # May Jun Jul for train, Aug Sep Oct for test
   # split_point = 22665
   # end_point =  32127

   # May Jun for train, Jul Aug Sep Oct for test
   # split_point = 8220 # upto Jul-12
   # mid_point = 9660 # upto Aug-18
   # end_point =  12777 # upto Oct-1
   
  #  split_point = 9210 # upto Jul-12
  #  mid_point = 10650 # upto Aug-18
  #  end_point =  13767 # upto Oct-1

   # split_point = 9210 # upto Jul-12
   # mid_point = 11187 # upto Sep-13
   # end_point =  13767 # upto Oct-1
   
   # this used 5s sliding window to augment the data
   # split_point = 109629 # upto Jul-12
   # mid_point = 133162 # upto Sep-13
   # end_point =  163959 # upto Oct-1 
   
   # test sliding window with less days of rain
   # split_point = 16545 # May 1
   # split_point = 19394 # upto May 2
   split_point = 27584 # upto May 11
   
   mid_point1 = 109629 # Sep-13
   mid_point2 = 133162 # upto Sep-13
   end_point =  163959 # upto Oct-1    

   
train = dataset.iloc[:split_point,:]
val = dataset.iloc[mid_point1:mid_point2, :]
test = dataset.iloc[mid_point2:end_point,:]

train.to_csv(output_path + 'train.csv', index=False)
val.to_csv(output_path + 'val.csv', index=False)
test.to_csv(output_path + 'test.csv', index=False)

print('Saved train val test .csv')