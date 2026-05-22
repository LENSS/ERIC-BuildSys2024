# -*- coding: utf-8 -*-
"""
Created on Sun May  9 21:54:25 2021

@author: Tian
"""

# combine audio and video features, get is_rain label
# generate csv file for training and testing

import os
import pandas as pd
#from datetime import datetime
#import matplotlib.pyplot as plt
import time
#from datetime import timedelta
import json
import yaml

#-----------------------------------------------------------------
def add_is_rain(fea, rain_minute_dict):
  
  is_rain = []
  rain_amount = []
  for t in fea['DateTime']:
    # minute_key = str(t)[:-3]
    minute_key = str(t) # actually second key for 5s label
    
    if minute_key in rain_minute_dict:
      is_rain.append(1)
      rain_amount.append(rain_minute_dict[minute_key])
    else:
      is_rain.append(0)
      rain_amount.append(0)
    
    # if second_key in rain_minute_dict:
    #   rain_amount.append(rainfall_dict[second_key])
    # else:
    #   rain_amount.append(0)
  
  fea['is_rain'] = is_rain
  fea['rain_amount'] = rain_amount
            
  return fea
  
#-----------------------------------------------------------------
def combine_audio_visual(audio, visual):
  """
  combine audio and visual feature based on same timestamps
  """
  res = pd.merge(audio,visual, on='DateTime', how='inner')
  return res

#-----------------------------------------------------------------
def split_day_night(data):
  """
  split based on is_day column
  """
  day = data[data['is_day'] != 0]
  night = data[data['is_day'] == 0]

  return day, night

#-----------------------------------------------------------------
def add_is_day(data):
  """
  add the is_day column to the feature matrix
  """

  is_day = []
  for r in range(data.shape[0]):
    # print("r =", r, "data['DateTime'][r] =", data['DateTime'][r])
    hour = data['DateTime'][r].hour
    if hour < 7 or hour >= 20:
      is_day.append(0)
    else:
      is_day.append(1)
  data['is_day'] = is_day
  res = data
      
  return res

#-----------------------------------------------------------------
def cal_avg(data, start, end, time_col = 'DateTime', interval = '0.5min'):
  """
  average the features every 30s
  """
  res = data.groupby(pd.Grouper(key='DateTime',freq = interval))[data.columns[start:end]].mean()
  res.reset_index(level=0, inplace=True) # reset the timelabel from index to column
  res.dropna(axis=0, inplace=True, how='any') # drop the rows with NAN values
  res.reset_index(drop=True, inplace=True)
  
  return res

def cal_avg_slidingwindow(data, start, end, time_col = 'DateTime', interval = '0.5min', hop = '5s', is_audio = False):
  """
  average the features every 30/60 seconds, but with a rolling window and hop length of 5s
  https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
  """

  # for audio features, we need to first groupby 5s then do rolling window
  if is_audio:
    data = data.groupby(pd.Grouper(key='DateTime',freq = '5s'))[data.columns[start:end]].mean()
    data.reset_index(level=0, inplace=True)
    data.dropna(axis=0, inplace=True, how='any')
    data.reset_index(drop=True, inplace=True)

  # print(data.tail())
  res = data.rolling(interval, center=False, on = time_col, min_periods = 1).mean()
  # res.dropna(axis=0, inplace=True, how='any')
  res.reset_index(drop=True, inplace=True)
  # print(res.tail())
  # stop  
  return res
    
#-----------------------------------------------------------------
def load_features(input_path, time_col = 'DateTime'):
  
  read_start = time.time()
  print('\nLoading features from: {}'.format(input_path))

  files = os.listdir(input_path)
  file_list = [i for i in files if i.endswith('.csv')]
  print("Total .csv found: {}".format(len(file_list)))
  file_list.sort()

  data_list=[]
  ct = 0
  
  for fn in file_list:
    if ct % 100 == 0 :
      print('{}/{}'.format(ct, len(file_list)), end = ' ')
  
    data = pd.read_csv(input_path + fn, parse_dates = [time_col])
    #print(data.info())

    data_list.append(data)
    ct += 1
  
  print()
  
  # merge all datasets
  df = pd.concat(data_list, ignore_index=True)

  # sort by the time
  df = df.sort_values(by = time_col, ascending = True)
  print('Before dropping duplicates: ', len(df))
  
  # removing duplicted rows with same DateTime label
  df = df.drop_duplicates(subset = time_col, keep = 'first')
  print('After dropping duplicates: ', len(df))
  print()
  print(df.info())

  read_end = time.time()
  read_time = round(read_end-read_start,2)
  print("\nFinish loading features, time = {} sec".format(read_time))

  return df

#---------------------------------------------------------------- main
with open('control.yaml') as f:
   controls = yaml.load(f, Loader=yaml.FullLoader)

input_visual_path = controls['input_visual_path']
input_audio_path = controls['input_audio_path']
output_path = controls['fea_path']
rain_label = controls['rain_label']

prefix = ''

if not os.path.isdir(output_path):
  os.mkdir(output_path)
  print("Folder {} created".format(output_path))
else:
  print("Folder {} already existed".format(output_path))

# load features
audio = load_features(input_audio_path)
visual = load_features(input_visual_path)

#---------- average features by 30s/1min interval
# audio_avg = cal_avg(audio, 0, 25, interval='1.0min')
# print(audio_avg.info())

# visual_avg = cal_avg(visual, 3, -1, interval='1.0min') # notice -1 here, last column is empty, thus ignored
# print(visual_avg.info())

#---------- average features by 1 min interval with sliding window of hop 5s
prefix = 'slidingwindow_'
audio_avg = cal_avg_slidingwindow(audio, 0, 25, interval='1.0min', hop = '5s', is_audio = True)
print(audio_avg.info())

visual_avg = cal_avg_slidingwindow(visual, 3, -1, interval='1.0min', hop = '5s') # notice -1 here, last column is empty, thus ignored
print(visual_avg.info())

#---------- add the is_day column
audio_avg = add_is_day(audio_avg)
visual_avg = add_is_day(visual_avg)

# combine day and night
fea_all = combine_audio_visual(audio_avg, visual_avg)

# only keep one column of is_day
fea_all['is_day']=fea_all['is_day_x']
fea_all.drop(['is_day_x','is_day_y'], axis=1, inplace=True)

# split into day and night
audio_avg_day, audio_avg_night = split_day_night(audio_avg)

visual_avg_day, visual_avg_night = split_day_night(visual_avg)
#print(audio_avg_night.describe())

#---------- combine audio and visual features
fea_day = combine_audio_visual(audio_avg_day, visual_avg_day)
fea_night = combine_audio_visual(audio_avg_night, visual_avg_night)

#----------- add is_rain, rainfall labels
with open(rain_label) as json_file:
  rain_minute_dict = json.load(json_file)

fea_day = add_is_rain(fea_day, rain_minute_dict)
fea_night = add_is_rain(fea_night, rain_minute_dict)
fea_all = add_is_rain(fea_all, rain_minute_dict)

# drop the is_day_x and is_day_y columns
fea_day.drop(['is_day_x','is_day_y'], axis=1, inplace=True)
fea_night.drop(['is_day_x','is_day_y'], axis=1, inplace=True)
#print(fea_day.describe())

# fea_day_rain = fea_day[fea_day['is_rain']==1]

#-------------------reorder the columns
fea_day_cols = \
['DateTime',
 'ZeroCrossingRate',
 'RMSE',
 'SpectralCentroids',
 'SpectralRolloff',
 'AmplitudeEnvelope',
 'MFCCS_1',
 'MFCCS_2',
 'MFCCS_3',
 'MFCCS_4',
 'MFCCS_5',
 'MFCCS_6',
 'MFCCS_7',
 'MFCCS_8',
 'MFCCS_9',
 'MFCCS_10',
 'MFCCS_11',
 'MFCCS_12',
 'MFCCS_13',
 'MFCCS_14',
 'MFCCS_15',
 'MFCCS_16',
 'MFCCS_17',
 'MFCCS_18',
 'MFCCS_19',
 'MFCCS_20',
 'max_1',
 'max_2',
 'max_3',
 'max_4',
 'max_5',
 'nonzero_pct_1',
 'nonzero_pct_2',
 'nonzero_pct_3',
 'nonzero_pct_4',
 'nonzero_pct_5',
 'variability_1',
 'variability_2',
 'variability_3',
 'variability_4',
 'variability_5',
 'highInt_pct_1',
 'highInt_pct_2',
 'highInt_pct_3',
 'highInt_pct_4',
 'highInt_pct_5',
 'is_rain',
 'rain_amount']

fea_all_cols = \
['DateTime',
 'is_day',
 'ZeroCrossingRate',
 'RMSE',
 'SpectralCentroids',
 'SpectralRolloff',
 'AmplitudeEnvelope',
 'MFCCS_1',
 'MFCCS_2',
 'MFCCS_3',
 'MFCCS_4',
 'MFCCS_5',
 'MFCCS_6',
 'MFCCS_7',
 'MFCCS_8',
 'MFCCS_9',
 'MFCCS_10',
 'MFCCS_11',
 'MFCCS_12',
 'MFCCS_13',
 'MFCCS_14',
 'MFCCS_15',
 'MFCCS_16',
 'MFCCS_17',
 'MFCCS_18',
 'MFCCS_19',
 'MFCCS_20',
 'max_1',
 'max_2',
 'max_3',
 'max_4',
 'max_5',
 'nonzero_pct_1',
 'nonzero_pct_2',
 'nonzero_pct_3',
 'nonzero_pct_4',
 'nonzero_pct_5',
 'variability_1',
 'variability_2',
 'variability_3',
 'variability_4',
 'variability_5',
 'highInt_pct_1',
 'highInt_pct_2',
 'highInt_pct_3',
 'highInt_pct_4',
 'highInt_pct_5',
 'is_rain',
 'rain_amount']

fea_all_noMFCCs_cols = \
['DateTime',
 'is_day',
 'ZeroCrossingRate',
 'RMSE',
 'SpectralCentroids',
 'SpectralRolloff',
 'AmplitudeEnvelope',
 'max_1',
 'max_2',
 'max_3',
 'max_4',
 'max_5',
 'nonzero_pct_1',
 'nonzero_pct_2',
 'nonzero_pct_3',
 'nonzero_pct_4',
 'nonzero_pct_5',
 'variability_1',
 'variability_2',
 'variability_3',
 'variability_4',
 'variability_5',
 'highInt_pct_1',
 'highInt_pct_2',
 'highInt_pct_3',
 'highInt_pct_4',
 'highInt_pct_5',
 'is_rain',
 'rain_amount']

fea_day = fea_day[fea_day_cols]
fea_night = fea_night[fea_day_cols]
fea_all = fea_all[fea_all_cols]

# output to csv file
print("\noutput .csv for classification and regression ")
fea_day.to_csv(output_path + prefix + 'fea_day.csv', index=False)
fea_night.to_csv(output_path + prefix + 'fea_night.csv', index=False)
fea_all.to_csv(output_path + prefix + 'fea_all.csv', index=False)

# remove the MFCCs will increase accuracy
MFCCs = ['MFCCS_{}'.format(i) for i in range(1,21)]
fea_all_noMFCCs = fea_all.drop(MFCCs, axis=1)
fea_all_noMFCCs = fea_all_noMFCCs[fea_all_noMFCCs_cols]

# rename the feature names
new_name_dict = dict()
for i in range(1,6):
   k = 'max_'+str(i)
   v = 'Max_int_chg_'+str(i)
   new_name_dict[k] = v

for i in range(1,6):
   k = 'highInt_pct_'+str(i)
   v = 'Brightness_'+str(i)
   new_name_dict[k] = v

for i in range(1,6):
   k = 'nonzero_pct_'+str(i)
   v = 'Density_'+str(i)
   new_name_dict[k] = v

for i in range(1,6):
   k = 'variability_'+str(i)
   v = 'Variability_'+str(i)
   new_name_dict[k] = v
   
fea_all_noMFCCs.rename(columns=new_name_dict, inplace=True)
fea_all_noMFCCs.to_csv(output_path + prefix + 'fea_all_noMFCCs.csv', index=False)

print('Complete!')
