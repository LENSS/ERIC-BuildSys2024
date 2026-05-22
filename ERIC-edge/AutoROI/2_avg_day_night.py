
# average day and night avg dfs

# import cv2
import matplotlib.pyplot as plt
import datetime as dt
import os.path
import os
import time
import getopt
import sys
import numpy as np
# import cProfile as profile

def plot_avg_gray_df(df, fea_path, case_name):

  fig, axs = plt.subplots(1,1,figsize=(10,10))
  im = axs.imshow(df, cmap='gray', vmin=0, vmax=1, norm = None)
  fig.colorbar(im, ax=axs)
  axs.set_title(case_name)
  plt.savefig(fea_path+case_name+'.png', bbox_inches='tight')
  #plt.close()

# TAMU_test
# npy_path_day = 'TAMU/20210815_day/Avg_Intensity/Avg_gray_df_0815_13-16pm.npy'
# npy_path_night = 'TAMU/20210601_night/Avg_Intensity/Avg_gray_df_0601_3-5am.npy'

# TAMU_train
# npy_path_day = 'TAMU_train/20210429_0430_0501/day/Avg_Intensity/Avg_gray_df.npy'
# npy_path_night = 'TAMU_train/20210429_0430_0501/night/Avg_Intensity/Avg_gray_df.npy'
# output_path = 'TAMU_train/'

# 2021 RaduHome 
# npy_path_day = '../../../2021_RaduHome_VideoData/AutoRoI/day/Avg_Intensity/Avg_gray_df.npy'
# npy_path_night = '../../../2021_RaduHome_VideoData/AutoRoI/night/Avg_Intensity/Avg_gray_df.npy'
# output_path = '../../../2021_RaduHome_VideoData/AutoRoI/'

# 2022 RaduHome
npy_path_day = '../../../Front/AutoRoI/day/Avg_Intensity/Avg_gray_df.npy'
npy_path_night = '../../../Front/AutoRoI/night/Avg_Intensity/Avg_gray_df.npy'
output_path = '../../../Front/AutoRoI/'

case_name = 'day_plus_night'

# load
day = np.load(npy_path_day)
night = np.load(npy_path_night)

avg = (day + night) / 2.0

np.save(output_path+case_name+'.npy', avg)
plot_avg_gray_df(avg, output_path, case_name)

