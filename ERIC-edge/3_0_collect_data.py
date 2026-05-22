"""
This scripts collects visual/audio files into a single file for train, val, tst 
"""

import os
import pandas as pd
#from datetime import datetime
#import matplotlib.pyplot as plt
import time
#from datetime import timedelta
import json
import numpy as np
import getopt
import sys

def add_rainlabel_aug(fea, rain_minute_dict_aug):
  is_rain = []
  rain_amount = []
  for t in fea['DateTime']:
    key = str(t) # this is key with finer resolution in seconds
    #print('key ={}|'.format(key))
    if key in rain_minute_dict_aug:
      is_rain.append(1)
      rain_amount.append(rain_minute_dict_aug[key])
      #print("found")
    else:
      is_rain.append(0)
      rain_amount.append(0)
      #print("no found")
    
    #stop
  
  fea['is_rain'] = is_rain
  fea['rain_amount'] = rain_amount
            
  return fea   

def add_rainlabel(fea, rain_minute_dict):
  is_rain = []
  rain_amount = []
  for t in fea['DateTime']:
    minute_key = str(t)[:-3]
    
    if minute_key in rain_minute_dict:
      is_rain.append(1)
      rain_amount.append(rain_minute_dict[minute_key])
    else:
      is_rain.append(0)
      rain_amount.append(0)
  
  fea['is_rain'] = is_rain
  fea['rain_amount'] = rain_amount
            
  return fea    


def combine_audio_visual(audio, visual):
  """
  combine audio and visual feature based on same timestamps
  """
  res = pd.merge(audio, visual, on='DateTime', how='inner')
  return res

def cal_avg(data, start, end, time_col = 'DateTime', interval = '1.0min'):
  """
  average the features every 60s
  """
  res = data.groupby(pd.Grouper(key='DateTime',freq = interval))[data.columns[start:end]].mean()
  res.reset_index(level=0, inplace=True) # reset the timelabel from index to column
  res.dropna(axis=0, inplace=True, how='any') # drop the rows with NAN values
  res.reset_index(drop=True, inplace=True)
  
  return res

def cal_avg_slidingwindow(data, start, end, time_col = 'DateTime', interval = '0.5min', hop = 5, is_audio = False):
  """
  average the features every 30/60 seconds, but with a rolling window and hop length of 5s
  https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
  """
  
  # for audio features, we need to first groupby 5s then do rolling window
  # for visual features, we also do groupby 5s to make the time consistent with audio features
  data = data.groupby(pd.Grouper(key='DateTime',freq = '5s'))[data.columns[start:end]].mean()
  data.reset_index(level=0, inplace=True)
  data.dropna(axis=0, inplace=True, how='any')
  data.reset_index(drop=True, inplace=True)
  # print("\nafter grouping\n", data.head())
  
  # print(data.tail())
  res = data.rolling(interval, center=False, on = time_col, min_periods = 1).mean()
  # res.dropna(axis=0, inplace=True, how='any')
  res.reset_index(drop=True, inplace=True)
  # print(res.tail())

  return res

def load_features(input_path, suffix):

    time_col = 'DateTime'
    read_start = time.time()
    print('\nLoading {} features from: {}'.format(suffix, input_path))

    files = os.listdir(input_path)
    file_list = [i for i in files if i.endswith(suffix)]
    print("Total {} found: {}".format(suffix, len(file_list)))
    file_list.sort()

    data_list=[]
    ct = 0
    
    for fn in file_list:
        if ct % 50 == 0 :
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
    # print(df.info())

    # for _visual.csv drop the last column
    if suffix == '_visual.csv':
        df = df.iloc[:, :-1]

    read_end = time.time()
    read_time = round(read_end-read_start,2)
    print("\nFinish loading features, time = {} sec".format(read_time))

    return df

#--------------------------------------main-------------
av_select_cols = [
 'DateTime',
 'is_Day',
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
 'rain_amount'
]

v_select_cols = [
 'DateTime',
 'is_Day',
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
 'rain_amount'
]

a_select_cols = [
'DateTime',
# 'is_Day',
'ZeroCrossingRate',
'RMSE',
'SpectralCentroids',
'SpectralRolloff',
'AmplitudeEnvelope',
'is_rain',
'rain_amount'
]

dataset = 0

try:
    options, remainder = getopt.getopt(sys.argv[1:], 'd:r:a:')
except getopt.GetoptError as err:
    print("getopt error: %s" % (str(err)))
    exit()

for opt, arg in options:
    if opt == '-d':
        input_dir = arg
    if opt == '-a':
        dataset = int(arg)

if dataset == 1: # 2021 RaduHome
  rain_label = '../process_rain/2021_RaduHome/output/2021_RaduHome_rainminute_annotated_interpolate-60s.json'
  rain_label_aug = '../process_rain/2021_RaduHome/output/2021_RaduHome_rainminute_annotated_interpolate-5s_slidingwindow.json'

elif dataset == 2: # 2021 TAMU west campus
  print("TAMU west campus not set!")
  exit(1)

elif dataset == 3: # 2022 Radu new Home
   rain_label = '../process_rain/2022_RaduHome/output/2022_RaduHome_rainminute_annotated_interpolate-60s.json'
   rain_label_aug = '../process_rain/2022_RaduHome/output/2022_RaduHome_rainminute_annotated_interpolate-5s_slidingwindow.json'

else:
  exit(1)


work_dir = input_dir + 'RS_ML/'

with open(rain_label) as json_file:
    rain_minute_dict = json.load(json_file)

with open(rain_label_aug) as json_file:
    rain_minute_dict_aug = json.load(json_file)

for folder in ['train/', 'val/', 'test/']:
    fea_path = work_dir + folder
    print('--'*10, fea_path)

    # collect all feature files into a dataframe
    visual = load_features(fea_path, '_visual.csv')
    audio = load_features(fea_path, '_audio.csv')

    print('visual_rows:', len(visual))
    print('audio_rows:', len(audio))

    # dropping empty audio rows (for case of missing audios)
    print("Before dropping empty rows: {} audio rows".format(len(audio)))
    audio_nan = audio.replace(0, np.nan)
    audio_clean = audio_nan.dropna(how='any', axis=0)
    print("After dropping empty rows: {} audio rows".format(len(audio_clean)))

    # average features by 1 min interval with sliding window of hop 5s
    if folder == 'train/':

        # prefix = 'slidingwindow_'
        audio_avg = cal_avg_slidingwindow(audio_clean, 0, 25, interval='1.0min', hop = 5, is_audio = True)
        visual_avg = cal_avg_slidingwindow(visual, 1, 23, interval='1.0min', hop = 1) # notice -1 here, last column is empty, thus ignored

        # combine the visual and audio data
        audio_visual = combine_audio_visual(audio_avg, visual_avg)

        # add is_rain and rain_amount column using augmentated rainlabels
        audio_visual = add_rainlabel_aug(audio_visual, rain_minute_dict_aug)
        visual_avg = add_rainlabel_aug(visual_avg, rain_minute_dict_aug)
        audio_avg = add_rainlabel_aug(audio_avg, rain_minute_dict_aug)

    else: # for val and tst, just groupby 1 minute, no augmentation by sliding window
        audio_avg = cal_avg(audio_clean, 0, 25, interval='1.0min')
        visual_avg = cal_avg(visual, 1, 23, interval='1.0min')

        # combine the visual and audio data
        audio_visual = combine_audio_visual(audio_avg, visual_avg)
    
        # add is_rain and rain_amount column
        audio_visual = add_rainlabel(audio_visual, rain_minute_dict)
        visual_avg = add_rainlabel(visual_avg, rain_minute_dict)
        audio_avg = add_rainlabel(audio_avg, rain_minute_dict)
    
    # select and reorder columns, here removed MFCCs
    audio_visual = audio_visual[av_select_cols]
    visual_avg = visual_avg[v_select_cols]
    audio_avg = audio_avg[a_select_cols]

    print(audio_visual.info())
    print(visual_avg.info())
    print(audio_avg.info())

    # dump out to csv file
    fn = work_dir+folder[:-1]+'_audiovisual.csv'
    audio_visual.to_csv(fn, index=False)
    print("Dumped", fn)

    fn = work_dir+folder[:-1]+'_visual.csv'
    visual_avg.to_csv(fn, index=False)
    print("Dumped", fn)

    fn = work_dir+folder[:-1]+'_audio.csv'
    audio_avg.to_csv(fn, index=False)
    print("Dumped", fn)