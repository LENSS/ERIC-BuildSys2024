
# cluster the avg to select boxes as Regions of Interest 

# import cv2
import matplotlib.pyplot as plt
import datetime as dt
import os.path
import os
import time
import getopt
import sys
import numpy as np
import pandas as pd
# import cProfile as profile
from sklearn.cluster import KMeans
from matplotlib.patches import Rectangle
import json


def get_box_cord(df, clus):
   
   # get the point in the same cluster
   print('\n---------- cluster', clus)
   data = df.loc[df['cluster'] == clus]
   num_points = len(data['cluster'])
   print('# of points:', num_points)
   
   x_cord_sorted = list(data['x_cord'])
   x_cord_sorted.sort()
   y_cord_sorted = list(data['y_cord'])
   y_cord_sorted.sort()
   
   # capture 80% of points in a cluster
   ul_x = int(x_cord_sorted[int(num_points*0.1)])
   ur_x = int(x_cord_sorted[int(num_points*0.9)])
   
   ul_y = int(y_cord_sorted[int(num_points*0.1)])
   bl_y = int(y_cord_sorted[int(num_points*0.9)])
   
   box = dict()
   box['ul'] = (ul_x, ul_y)
   box['ur'] = (ur_x, ul_y)
   box['bl'] = (ul_x, bl_y)
   box['br'] = (ur_x, bl_y)
   
   box['x_len'] = ur_x - ul_x
   box['y_len'] = bl_y - ul_y
   
   for k, v in box.items():
      print(k, v)
   
   return box
      

def plot_avg_gray_df(df, fea_path, case_name):

  fig, axs = plt.subplots(1,1,figsize=(10,10))
  im = axs.imshow(df, cmap='gray', vmin=0, vmax=3, norm = None)
  fig.colorbar(im, ax=axs)
  axs.set_title(case_name)
  plt.savefig(fea_path+case_name+'.png', bbox_inches='tight')
  # plt.close()

# TAMU_test
# npy_path = 'TAMU/day_plus_night.npy'

# npy_path = 'TAMU/20210815_day/Avg_Intensity/Avg_gray_df_0815_13-16pm.npy'
# npy_path = 'TAMU/20210601_night/Avg_Intensity/Avg_gray_df_0601_3-5am.npy'

# TAMU_train
# npy_path = 'TAMU_train/day_plus_night.npy'

# Radu home
# npy_path = 'Radu/Avg_Intensity/Avg_gray_df.npy'

# AAU
# npy_path = 'AAU/Avg_Intensity/Avg_gray_df.npy'

# Other
# npy_path = 'Other/Avg_Intensity/Avg_gray_df.npy'

# TAMU train all data
# npy_path = 'TAMU_train_all/Avg_Intensity/Avg_gray_df_alltrain.npy'

# 2022 radu home data
#npy_path = '20221024_radu_home_back/Avg_Intensity/Avg_gray_df_2.npy'
# npy_path = '20221024_radu_home_front/Avg_Intensity/Avg_gray_df_1.npy'


#------ 2021 RaduHome 
# npy_path = '../../../2021_RaduHome_VideoData/AutoRoI/day_plus_night.npy'
# threshold = 0.15 # 0.15
# num_clus = 5

#------ 2022 RaduHome
npy_path = '../../../Front/AutoRoI/day_plus_night.npy'
threshold = 0.08 # 0.15
num_clus = 5


output_path = '/'.join(npy_path.split('/')[:-1])+'/'
case_name = 'Avg_gray_df_tsh{}'.format(threshold)


#---------------------- clustering for RoI
# load
avg = np.load(npy_path)
# print(avg.mean())

# convert to data points
dim1, dim2 = avg.shape
x_cord = []
y_cord = []
inten_val = []

# for 2022 radu home, filter out up and down wall reflections (white background)
#avg[:500,:] = 0
#avg[1500:,:] = 0

for i in range(dim1):
  for j in range(dim2):
    if avg[i,j] > threshold: # filter out small intensity changes
      x_cord.append(i)
      y_cord.append(j)
      inten_val.append(avg[i,j])

print("pct = ", len(inten_val)/(dim1*dim2))

data_dict = {'x_cord':x_cord, 'y_cord':y_cord, 'inten_val':inten_val}
df = pd.DataFrame(data_dict)
print(df.describe())

# use weighted k-means clustering
kmeans = KMeans(n_clusters=num_clus, init='k-means++', random_state=100, max_iter=1000)
data = df.iloc[:,:2].values

weight = df['inten_val'].values
# weight = [1 for i in range(len(inten_val))] # if equal weights

wt_kmeansclus = kmeans.fit(data, sample_weight = weight)
predicted_kmeans = kmeans.predict(data, sample_weight = weight)

# visualize clusters
plt.style.use('default')
f, ax = plt.subplots(figsize=(6.5, 13))
ax.scatter(data[:,1], data[:,0], c=wt_kmeansclus.labels_.astype(float), s=1, cmap='tab20b', marker='x', )
ax.set_xlim([0, 1536])
ax.set_ylim([0.01, 2048])
ax.set_aspect('equal', adjustable='box')
#plt.set(xlim=(0, 1536), ylim=(0.01, 2048))


plt.title('Weighted K-Means (threshold = {})'.format(threshold), fontweight='bold')
plt.xlabel('y_cord')
plt.ylabel('x_cord')
centers = wt_kmeansclus.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=100, alpha=0.5)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_path+case_name+'_clusters.png', bbox_inches='tight')

# label each datapoint with its cluster number
df['cluster'] = predicted_kmeans

# define boxes
box_cord_dict = dict()
for clus in range(num_clus):
   box_cord = get_box_cord(df, clus)
   box_cord_dict[clus] = box_cord

# output boxes coordinates
with open(output_path + "boxes.json", "w") as outfile:
    json.dump(box_cord_dict, outfile, indent = 4)
print('\nDumped boxes coordinates.')

# draw the boxes
for clus, box in box_cord_dict.items():
   
   # flip the x y coordinate to ensure consistency as image
   anchor = (box['ul'][1], box['ul'][0])
   x_len = box['y_len']
   y_len = box['x_len']
   

   plt.gca().add_patch(Rectangle(anchor,x_len,y_len,
                    edgecolor='red',
                    facecolor='none',
                    lw=4))

plt.savefig(output_path+case_name+'_clusters_boxes.png', bbox_inches='tight')

#plot_avg_gray_df(avg, output_path, case_name)

