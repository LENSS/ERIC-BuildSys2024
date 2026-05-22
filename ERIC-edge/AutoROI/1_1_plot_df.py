import matplotlib.pyplot as plt
import datetime as dt
import sys
import numpy as np

def plot_avg_gray_df(df, fea_path, case_name):

  fig, axs = plt.subplots(1,1,figsize=(10,10))
  im = axs.imshow(df, cmap='gray', vmin=0, vmax=1, norm = None)
  fig.colorbar(im, ax=axs)
  axs.set_title(case_name)
  plt.savefig(fea_path+case_name+'.png', bbox_inches='tight')
  plt.close()


#npy_path = '20221024_radu_home_back/Avg_Intensity/Avg_gray_df.npy'
#feature_path = '20221024_radu_home_back/Avg_Intensity/'

npy_path = '20221024_radu_home_front/Avg_Intensity/Avg_gray_df.npy'
feature_path = '20221024_radu_home_front/Avg_Intensity/'

case_name = 'Avg_gray_df'

avg_gray_df = np.load(npy_path) # load
plot_avg_gray_df(avg_gray_df, feature_path, case_name)