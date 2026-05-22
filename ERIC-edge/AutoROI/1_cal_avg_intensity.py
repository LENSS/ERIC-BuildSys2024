# -*- coding: utf-8 -*-

# calculate the cumulative average pseudo-rain streaks over a period of time 
# to identify the regions of interest for rain visual feature calculation

# To do: can rescale the frame to reduce computation cost

import cv2
import matplotlib.pyplot as plt
import datetime as dt
import os.path
import os
import time
import getopt
import sys
import numpy as np
import cProfile as profile

#------------------------------------------------------------------------------
def print_fea(feature):

  for label in label_list:
    print(label)
    for i in range(0, grid_total):
      print(i)
      for x in range(len(feature[i][label])):
        print("{:.4f}".format(feature[i][label][x]), end =" ")
      print("\n","--"*30)
  print("--"*80)

#------------------------------------------------------------------------------
def plot_img(img):
  fig, axs = plt.subplots(1,1,figsize=(8,8))
  axs.imshow(img[:,:,::-1])
  plt.show()
  plt.close()

#------------------------------------------------------------------------------
def plot_grayimg(grayimg, fn, frame_id=None, save=False, badframe_path = None):

  fig, axs = plt.subplots(1,1,figsize=(10,10))
  im = axs.imshow(grayimg, cmap='gray', vmin=0, vmax=3, norm = None)
  fig.colorbar(im, ax=axs)

  if frame_id:
    axs.set_title("gray_img_"+str(frame_id-1)+'_'+str(frame_id))

  if save:
      plt.savefig(badframe_path + fn + "_bad_frame_"+str(frame_id-1)+'_'+str(frame_id), bbox_inches='tight')
  else:
      plt.show()

  plt.close()

#------------------------------------------------------------------------------
def cal_feature1(dfc):

  """
  old version
  calculate the features of each grid, 
  including min, max, nonzero_pct, variability, 401_ct, 412_ct, 423_ct, 434_ct
  """
  ftc={}

  (dim1,dim2) = dfc.shape
  total_ct = dim1 * dim2

  # variability, min, max, nonzero_ct
  var={}
  imax=0
  nonzero_ct=0

  for i in range(dim1):
    for j in range(dim2):
      
      if dfc[i,j] <= zero_intensity:
        continue # skip very low intensity change pixels
      else:
        nonzero_ct += 1
        imax = max(imax, dfc[i,j])

      var_bin = int(dfc[i,j]) # already skipped 0 intensity cells
      if var_bin not in var.keys():
        var[var_bin] = 1
      else:
        var[var_bin] += 1

  ftc["max"] = imax
  nonzero_pct = nonzero_ct / total_ct
  ftc["nonzero_pct"] = nonzero_pct

  variability = len(var)
  ftc["variability"] = variability

  # use fixed value bound
  imax1=3
  imax2=5
  imax3=10
  imax4=15

  imax1_ct=0
  imax2_ct=0
  imax3_ct=0
  imax4_ct=0
  
  for k, val in var.items():
    if k < imax1:
      imax1_ct += val
    elif k < imax2:
      imax2_ct += val
    elif k < imax3:
      imax3_ct += val
    else:
      imax4_ct += val
    
  ftc["imax1"] = imax1
  ftc["imax2"] = imax2
  ftc["imax3"] = imax3
  ftc["imax4"] = imax4

  ftc["imax1_ct"] = imax1_ct
  ftc["imax2_ct"] = imax2_ct
  ftc["imax3_ct"] = imax3_ct
  ftc["imax4_ct"] = imax4_ct

  return ftc

#------------------------------------------------------------------------------
def cal_feature2(dfc):

  """
  new version: less features for faster calculation
  calculate the features of each grid, 
  including max, nonzero_pct, variability, high_intensity_pct
  """
  ftc={}

  (dim1,dim2) = dfc.shape
  total_ct = dim1 * dim2

  var = {}
  #imax = 0
  nonzero_ct = 0

  for i in range(dim1):
    for j in range(dim2):
      if dfc[i,j] <= zero_intensity:
        continue # skip no intensity change pixels
      else:
        nonzero_ct += 1
        #imax=max(imax,dfc[i,j])

        inten = int(dfc[i,j])
        if inten not in var.keys():
          var[inten] = 1
        else:
          var[inten] += 1

  if len(var) == 0:
    ftc["max"] = 0
  else:
    ftc["max"] = max(var.keys())
  
  #print('max =', ftc["max"])

  ftc["nonzero_pct"] = nonzero_ct / total_ct
  ftc["variability"] = len(var)

  highInt_ct = 0
  for k, val in var.items():
    if k >= high_intensity:
      highInt_ct += val

  ftc["highInt_pct"] = highInt_ct / total_ct

  return ftc

#------------------------------------------------------------------------------
def cal_nonzero_pct(df):

  nz_ct = 0
  for i in range(df.shape[0]):
    for j in range(df.shape[1]):
      if df[i,j] > zero_intensity:
        nz_ct += 1

  nz_pct = nz_ct / (df.shape[0]*df.shape[1])

  return nz_pct

#------------------------------------------------------------------------------
def get_feature(df):

  """
  Divide the grayimage into grids, then calculate the features of each grid
  """
  accept = True

  dfc=[[] for i in range(grid_total)]
  (dim1, dim2, dim3) = df[0].shape # row, col

  if grid_flag == 0: # door + sky + ground, for Radu home
    dfc[0]=df[300:1700,350:1050]
    dfc[1]=df[300:900, 350:1050]
    dfc[2]=df[1120:1600,350:1050]
  
  elif grid_flag == 1: # for full + 3x3 grid
    dfc[0]=df #full image
    
    dfc[1]=df[0:int(dim1/3),0:int(dim2/3)]
    dfc[2]=df[0:int(dim1/3),int(dim2/3):int(2*dim2/3)]
    dfc[3]=df[0:int(dim1/3),int(2*dim2/3):dim2]

    dfc[4]=df[int(dim1/3):int(2*dim1/3),0:int(dim2/3)]
    dfc[5]=df[int(dim1/3):int(2*dim1/3),int(dim2/3):int(2*dim2/3)]
    dfc[6]=df[int(dim1/3):int(2*dim1/3),int(2*dim2/3):dim2]

    dfc[7]=df[int(2*dim1/3):dim1,0:int(dim2/3)]
    dfc[8]=df[int(2*dim1/3):dim1,int(dim2/3):int(2*dim2/3)]
    dfc[9]=df[int(2*dim1/3):dim1,int(2*dim2/3):dim2]

  elif grid_flag == 2 : # single grid + sky + building, for AAU crossing1 data only
    dfc[0] = df
    dfc[1] = df[0:280,:]
    dfc[2] = df[280:,:]

  elif grid_flag == 3 : # TAMU west campus
    dfc[0] = df

    # dfc[1] = df[740:900,0:360] # black window/board, inclduing the left light
    # dfc[2] = df[1050:1200,0:200] # main sidewalk
    # dfc[3] = df[1050:1100,200:800] # horizontal sidewalk
    # dfc[4] = df[700:850,650:1050] # mid light
    # dfc[5] = df[850:900,1395:1485] # right light

    dfc[1] = abs(df[1][800:920,0:360].mean(axis=2) - df[0][800:920,0:360].mean(axis=2))  # black window/board, inclduing the left light
    dfc[2] = abs(df[1][1040:1160,0:300].mean(axis=2) - df[0][1040:1160,0:300].mean(axis=2)) # left sidewalk
    dfc[3] = abs(df[1][1050:1080,200:800].mean(axis=2) - df[0][1050:1080,200:800].mean(axis=2))# horizontal sidewalk
    dfc[4] = abs(df[1][710:940,420:920].mean(axis=2) - df[0][710:940,420:920].mean(axis=2)) # tree, including midlight
    dfc[5] = abs(df[1][1300:1600,600:1000].mean(axis=2) - df[0][1300:1600,600:1000].mean(axis=2)) # water pond on the grass

  elif grid_flag == -1: # single gird, entire frame
    dfc[0] = abs(df[1][:,:].mean(axis=2) - df[0][:,:].mean(axis=2))

  # calculate features for each grid
  ft = [{} for i in range(grid_total)]

  # check the full image first to skip the shaking case
  full_nonzero_pct = cal_nonzero_pct(dfc[0])

  if full_nonzero_pct > 0.3:
    accept = False
    return accept, ft
  
  """
  # else cal other grids
  for i in range(1, grid_total):   
    ft[i] = cal_feature2(dfc[i]) # +++++++++ calculate features

    # check other grids as well
    '''
    if ft[i]["nonzero_pct"] > nonzero_pct_threshold:
      accept = False
      return accept, ft
    '''
  """

  return accept, ft

#------------------------------------------------------------------------------
def write_data(fw, feature, frame_id, time, is_day):
  """
  write the calculated features of each video to its corresponding file
  """
  fw.write(str(frame_id)); fw.write(',')
  t = str(time.year) + "-" + str(time.month) + "-" + str(time.day) + " " + str(time.hour) + ":" + str(time.minute) + ":" + str(time.second)
  fw.write(t); fw.write(",")
  fw.write(str(is_day)); fw.write(",")

  for label in label_list:
    for i in range(1, grid_total):
      #calculate the avg for each label of each grid

      #throw out the largest and smallest values in the list
      val = np.array(feature[i][label])
      #val = list(feature[i][label])

      #val.remove(max(mylist)) # remove the largest
      #val.remove(min(val)) # remove the smallest
      
      if val.shape[0] == 0:
        avg = 0
      else:
        avg = np.mean(val)

      fw.write(str(avg)); fw.write(",")

  fw.write('\n')
  fw.flush() 

#------------------------------------------------------------------------------
def write_single_data(fw, ft, frame_id, time, is_day):
  """
  write a single data point for each sampling frequency
  """
  fw.write(str(frame_id)); fw.write(',')
  t = str(time.year) + "-" + str(time.month) + "-" + str(time.day) + " " + str(time.hour) + ":" + str(time.minute) + ":" + str(time.second)
  fw.write(t); fw.write(",")
  fw.write(str(is_day)); fw.write(",")

  for label in label_list:
    for i in range(1, grid_total):
      fw.write("{:.4f}".format(ft[i][label])); fw.write(",")

  fw.write('\n')

#------------------------------------------------------------------------------
def get_time_from_filename(fn):
  """
  parse the time from the video filename

  -----for Radu doorbell videos:
  fn = 'doorbell-20201219-125305.mp4'
  -----for AAU videos:
  fn = '20180428_130116_795D-comb-cfr-cfr-left.mp4'
  -----for westcampus:
  fn = '2021-02-25T15-00-01.mp4'

  """
  if grid_flag == 2:
    time = dt.datetime.strptime(fn[:15],"%Y%m%d_%H%M%S")

  elif grid_flag == 3: # west campus
    fn_m = fn[:17] + "00.mp4" # fix the start second to 00
    time = dt.datetime.strptime(fn_m,"%Y-%m-%dT%H-%M-%S.mp4")

  else:
    time = dt.datetime.strptime(fn[:24],"doorbell-%Y%m%d-%H%M%S")

  t = str(time.year)+"-"+str(time.month)+"-"+str(time.day)+" "+str(time.hour)+":"+str(time.minute)+":"+str(time.second)

  print("video starting time: ",t)

  return time


def get_frame_dim(video_path):

  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  print("FPS = ", round(fps,4))
  # fps = 30
  # print("Set FPS = 30")

  cap.set(cv2.CAP_PROP_POS_FRAMES, 14)
  success, frame = cap.read()
  if not success:
    print("error: fail to read the frame")    
  print(frame.shape)
  dim1, dim2, dim3 = frame.shape
  print('Frame dimension: {}x{}'.format(dim1, dim2))

  return dim1, dim2


def cut_high_intensity(df):
  r, c = df.shape
  res = np.zeros((r,c))

  for rr in range(r):
    for cc in range(c):
      if df[rr,cc] > 3:
        res[rr, cc] = 3
      else:
        res[rr, cc] = df[rr, cc]

  return res

#------------------------------------------------------------------------------
def main(cal=True, plot=True):
  """
  $ python 1_cal_avg_intensity.py -a 1 -d ../../../2021_RaduHome_VideoData/AutoRoI/day/

  -a: dataset, 1, 2, 3
              # 1: 2021 RaduHome
              # 2: 2021 TAMU
              # 3: 2022 RaduHome
              
  -d: video_dir
  -f: feature_folder
  -s: video_start (inclusive, from 1)
  -e: video_end (inclusive, from 1)
  -c: case_name
  """

  _start = time.time()
  # default
  video_dir = './'
  feature_folder = 'Avg_Intensity/'
  case_name = 'Avg_gray_df'

  try:
      options, remainder = getopt.getopt(sys.argv[1:], 'a:d:f:s:e:c:')
  except getopt.GetoptError as err:
      print("getopt error: %s" % (str(err)))
      return

  for opt, arg in options:
    if opt == '-a':
      dataset = int(arg)
    if opt == '-d':
      video_dir = arg
    if opt == '-f':
      feature_folder = arg
    if opt == '-c':
      case_name = arg

  print('dataset =', dataset)
  print('video_dir =', video_dir)
  print('feature_folder =', feature_folder)
  print('case_name =', case_name)

  # list all .mp4 files
  files = os.listdir(video_dir)
  fn_list = [i for i in files if i.endswith('.mp4')]
  print("\nTotal # of Video Files: ", len(fn_list))
  fn_list.sort()
  for i in range(len(fn_list)):
    print(i+1, fn_list[i])
  print()

  # default
  video_start = 1
  video_end = len(fn_list)

  for opt, arg in options:
    if opt == '-s':
      video_start = int(arg)
    if opt == '-e':
      video_end = int(arg)
  
  print('len(fn_list) =', len(fn_list))  
  print('video_start = ', video_start)
  print('video_end (inclusive from 1) = ', video_end)

  #i_freq = smp_freq * 2 * avg_freq # control averaging
  current_video_idx = video_start

  # create the feature folder
  feature_path = video_dir + feature_folder
  print('feature_path =', feature_path)

  if not os.path.isdir(feature_path):
    os.mkdir(feature_path)
    print("Folder {} created".format(feature_path))
  else:
    print("Folder {} already existed".format(feature_path))

  """
  # create badframe folder
  badframe_path = feature_path + 'BadFrames/'
  if not os.path.isdir(badframe_path):
    os.mkdir(badframe_path)
    print("Folder {} created".format(badframe_path))
  else:
    print("Folder {} already existed".format(badframe_path))
  """

  npy_path = feature_path + case_name + '.npy'

  if cal:

    accept_ct = 0

    # get the frame size to init
    gray_df_lst = []
    
    fn_list = fn_list[video_start-1 : video_end]

    dim1, dim2 = get_frame_dim(video_dir + fn_list[0])
    cumu_gray_df = np.zeros((dim1, dim2))

    for fn in fn_list: # loop through each video file

      print()
      print("-"*10 + "Reading video file:", str(current_video_idx) + '/' + str(len(fn_list)), fn)
      process_start = time.time()

      # Read the video file info
      cap = cv2.VideoCapture(video_dir + fn)
      fps = cap.get(cv2.CAP_PROP_FPS)
      print("FPS = ", round(fps,4))
      
      fps = 30
      #fps = 22
      # print("Set FPS = 30")

      frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
      frame_end = frame_count - 1
      print("Total # of frames = ", frame_count)
      duration = frame_count / fps / 60 # in minute
      print("Video duration = {:.2f} minutes\n".format(duration))
      
      # Initialization
      i = 0
      #feature=[{} for g in range(grid_total)]
      frame_id = 15 # FPS = 30, take middle
      
      # for 2022 Radu home front, day, 2022-10-28-102743.mp4
      # frame_start = 13*60*30
      # frame_id = frame_start
      # frame_end = 25*60*30

      # for 2022 Radu home front, 2022-11-04-220519.mp4
      frame_start = 12*60*30
      # frame_id = frame_start
      # frame_end = 16*60*30


      #read_start = time.time()
      
      while(cap.isOpened() and frame_id <= frame_end) : # loop through the frames
    
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id-1)
        success, frame1 = cap.read()
        if not success:
          print("error: fail to read the frame", frame_id-1)
          break
        else: 
          i += 1
      
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame2 = cap.read()
        if not success:
          print("error: fail to read the frame", frame_id)
          break
        else: 
          i += 1
        
        #print('type(frame1) =',type(frame1))
        #print('shape =', frame1.shape)
        gray_f1 = frame1.mean(axis=2)
        gray_f2 = frame2.mean(axis=2)
        gray_df = abs(gray_f2 - gray_f1)

        # balck out the time label
        if dataset == 1 or dataset == 2: # 2021 RaduHome or 2021 TAMU West Campus
          gray_df[1776:1811,480:1050] = 0
        elif dataset == 3: # 2022 RaduHome
          gray_df[1909:1950, 65:640] = 0
        
        # crop out the caption
        # plot_img(frame1[1776:1811,480:1050,:])
        # gray_df[1776:1811,480:1050] = 0
        #plot_grayimg(gray_df, fn)
        
        """
        frame_pair = [frame1, frame2]
        accept, ft = get_feature(frame_pair) # calculate the visaul features
        
        if accept:
          cumu_gray_df += gray_df
          accept_ct += 1   
        """

        # normalize the gray_df to avoid single df with large influence
        # max_df_value = np.max(gray_df)
        # mean_df_value = np.mean(gray_df)

        #gray_df_cut = cut_high_intensity(gray_df)
        #cumu_gray_df += gray_df_cut

        cumu_gray_df += gray_df
        accept_ct += 1      

        if i % 120 == 0: # show progress every 5 minutes for 5s interval        
          print("{}".format(round(frame_id/frame_count, 2)), end=" ", flush = True)

        frame_id += (fps * sample_freq) # FPS x seconds

      print("\n" + "-"*10 + "Total frames processed: ", i, "in", fn)
      process_end = time.time()
      process_time = (process_end - process_start) / 60
      print("-"*10 + "Processing time: ", round(process_time,2), "mins")
      current_video_idx += 1
      cap.release()
    
    # avg all the gray_dfs
    avg_gray_df = cumu_gray_df / accept_ct
    
    print('npy_path =', npy_path)
    np.save(feature_path+case_name+'.npy', avg_gray_df) # save
    print("\nAvg_gray_df.npy saved!")
    
    _end = time.time()
    total_time = (_end - _start) / 3600
    print("\n"+"="*10 + "Finished all videos, total processing time: ", round(total_time,2), "hrs ")

  if plot:
    avg_gray_df = np.load(npy_path) # load
    plot_avg_gray_df(avg_gray_df, feature_path, case_name)


def plot_avg_gray_df(df, fea_path, case_name):

  fig, axs = plt.subplots(1,1,figsize=(10,10))
  im = axs.imshow(df, cmap='gray', vmin=0, vmax=1, norm = None)
  fig.colorbar(im, ax=axs)
  axs.set_title(case_name)
  plt.savefig(fea_path+case_name+'.png', bbox_inches='tight')
  plt.close()

#-----------------------------------------------------------------
if __name__ == "__main__":

  # global
  sample_freq = 5 # sample 1 df every 5s
  grid_flag = -1
  grid_total = 1
  zero_intensity = 0.1

  #profile.runctx('main()', globals(),locals(), sort='cumtime')
  main(cal=True, plot=True)
  #main(cal=False, plot=True)