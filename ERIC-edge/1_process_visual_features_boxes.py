# -*- coding: utf-8 -*-
"""
Created on Sun May  9 00:40:20 2021

@author: Tian
"""

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
import json

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
        continue # skip small intensity change pixels
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
def get_feature(df, boxes):

  """
  Divide the grayimage into grids, then calculate the features of each grid
  """
  accept = True

  df_RoI=[]
  for k, box in boxes.items():
    x0 = box['ul'][0]
    x1 = box['ur'][0]
    y0 = box['ul'][1]
    y1 = box['bl'][1]
    # print('k =', k, 'x0 =', x0, 'x1 =', x1, 'y0 =', y0, 'y1 =', y1)
    # print('df[0].shape =', df[0].shape, 'df[1].shape =', df[1].shape)

    df_box = abs(df[1][x0:x1+1, y0:y1+1].mean(axis=2) - df[0][x0:x1+1, y0:y1+1].mean(axis=2))
    df_RoI.append(df_box)

  # calculate features for each grid
  ft = []

  # check the full image first to skip the shaking case
  # full_nonzero_pct = cal_nonzero_pct(dfc[0])

  # if full_nonzero_pct > nonzero_pct_threshold:
  #   accept = False
  #   return accept, ft
    
  # else cal other grids
  for idx, df_box in enumerate(df_RoI):
    # print('idx =', idx, 'df_box.shape =', df_box.shape)   
    ft_box = cal_feature2(df_box) # +++++++++ calculate features
    ft.append(ft_box)

    # check other grids as well
    '''
    if ft[i]["nonzero_pct"] > nonzero_pct_threshold:
      accept = False
      return accept, ft
    '''

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
    for i in range(0, grid_total):
      fw.write("{:.4f}".format(ft[i][label])); fw.write(",")

  fw.write('\n')

#------------------------------------------------------------------------------
def get_time_from_filename(fn):
  """
  parse the time from the video filename
  """
  
  if dataset == 1: # 2021 RaduHome, 'doorbell-20201219-125305.mp4'
    time = dt.datetime.strptime(fn[:24],"doorbell-%Y%m%d-%H%M%S")

  elif dataset == 2: # 2021 TAMU West Campus, '2021-02-25T15-00-01.mp4'
    fn_m = fn[:17] + "00.mp4" # fix the start second to 00
    time = dt.datetime.strptime(fn_m,"%Y-%m-%dT%H-%M-%S.mp4")
    
  elif dataset == 3: # 2022 RaduHome
    if fn[:5] == 'Front': # 'Front-2022-10-17-035243.mp4'
      time = dt.datetime.strptime(fn[6:23],"%Y-%m-%d-%H%M%S")

    elif fn[:4] == 'Back': # 'Back-20221024-230315.mp4'
      time = dt.datetime.strptime(fn[5:22],"%Y-%m-%d-%H%M%S")

  elif dataset == 4: # AAU, '20180428_130116_795D-comb-cfr-cfr-left.mp4'
    time = dt.datetime.strptime(fn[:15],"%Y%m%d_%H%M%S")

  t = str(time.year)+"-"+str(time.month)+"-"+str(time.day)+" "+str(time.hour)+":"+str(time.minute)+":"+str(time.second)

  print("video starting time: ",t)

  return time

#------------------------------------------------------------------------------
def main():
  """
  $ python 1_process_visual_features_boxes.py -a 1 -d ../../2021_RaduHome_VideoData/Rain_Audio/ -b ../../2021_RaduHome_VideoData/AutoRoI/boxes.json

  -a: dataset, 1, 2, 3
              # 1: 2021 RaduHome
              # 2: 2021 TAMU
              # 3: 2022 RaduHome
              # 4: AAU Dataset
  -d: video_dir, this is the parent dir containing many subfolders
  -f: feature_folder
  -s: video_start (inclusive, from 1)
  -e: video_end (inclusive, from 1)
  -b: path to boxes.json, which defines regions of interest
  """

  _start = time.time()
  # default
  video_dir = './'
  feature_folder = 'VisualFeatures/'
  box_path = None
  global dataset

  try:
      options, remainder = getopt.getopt(sys.argv[1:], 'a:d:f:s:e:b:')
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
    if opt == '-b':
      box_path = arg

  print('video_dir =', video_dir)
  print('feature_folder =', feature_folder)
  print('box_path =', box_path)

  # fetch boxes.json for RoI
  if box_path is not None:
    with open(box_path) as f:
      boxes = json.load(f)

    global grid_total
    grid_total = len(boxes)
    print('\nLoaded {} boxes (RoIs)'.format(len(boxes)))
  else:
    print('boxes.json missing!')
    raise ValueError()
  
  # list all .mp4 files
  subfolders = os.listdir(video_dir)
  # subfolders = glob.glob("*/")
  print('# of subfolders in current parent dir:', len(subfolders))
  subfolders.sort()
  for i in range(len(subfolders)):
    print(i+1, subfolders[i])  

  fn_list = []
  for subf in subfolders:
    files = os.listdir(video_dir+subf)
    fns = [video_dir+subf+'/'+i for i in files if i.endswith('.mp4')]        
    fns.sort()
    fn_list.extend(fns)

  for i in range(len(fn_list)):
    print(i+1, fn_list[i])

  print("# of Video Files: ", len(fn_list))
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
  print('video_end (inclusive) = ', video_end)

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

  # create badframe folder
  badframe_path = feature_path + 'BadFrames/'
  if not os.path.isdir(badframe_path):
    os.mkdir(badframe_path)
    print("Folder {} created".format(badframe_path))
  else:
    print("Folder {} already existed".format(badframe_path))

  # create the log file
  log_filepath = feature_path + 'VisualProcessor.log'
  fwlog = open(log_filepath,'w')

  for fn in fn_list[video_start-1 : video_end]: # loop through each video file

    print()
    print("-"*10 + "Reading video file:", str(current_video_idx) + '/' + str(len(fn_list)), fn)
    process_start = time.time()

    # create feature file
    fn1 = fn.split('/')[-1].split(sep = '.')[0]
    feature_filepath = feature_path + fn1 + '_visual.csv'
    fw = open(feature_filepath, 'w')
    fw.write('FrameID'); fw.write(",")
    fw.write('DateTime'); fw.write(",")
    fw.write('is_Day'); fw.write(",")

    for label in label_list: # write the label title
      for grid_i in range(1, grid_total+1):
        fw.write(label + '_' + str(grid_i))
        fw.write(",")
    fw.write('\n')
    fw.flush()

    # Read the video file info
    cap = cv2.VideoCapture(fn)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS = ", round(fps,4))
    fps = 30
    print("Set FPS = 30")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total # of frames = ", frame_count)
    duration = frame_count / fps / 60 # in minute
    print("Video duration = {:.2f} minutes\n".format(duration))
    
    # Initialization
    i = 0
    time_label = get_time_from_filename(fn.split('/')[-1])
    frame_id = 15 # FPS = 30, take middle
  
    while(cap.isOpened() and frame_id <= frame_count) : # loop through the frames

      th = time_label.hour
      if th < 7 or th >= 20: # light turns off at 7am, turns on at 8pm
        is_day = 0
      else:
        is_day = 1
  
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id-1)
      success, frame1 = cap.read()
      if not success:
        print("error: fail to read the frame", frame_id -1)
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
      
      # balck out the time label caption
      if dataset == 1 or dataset == 2: # 2021 RaduHome or 2021 TAMU West Campus
        frame1[1776:1811, 480:1050, :] = 0
        frame2[1776:1811, 480:1050, :] = 0

      elif dataset == 3: # 2022 RaduHome
        frame1[1909:1950, 65:640, :] = 0
        frame2[1909:1950, 65:640, :] = 0
      
      gray_df = [frame1, frame2]
      accept, ft = get_feature(gray_df, boxes) # calculate the visaul features +++++

      if accept:
        write_single_data(fw, ft, frame_id, time_label, is_day)
      else:
        print("\nwarning: skip bad frame ({},{})".format(frame_id-1, frame_id))
        if save_badframe:
          plot_grayimg(gray_df, fn1, frame_id, True, badframe_path)
        fwlog.write("\nwarning: skip bad frame ({},{})".format(frame_id-1, frame_id))
        fwlog.flush()

      if i % 120 == 0: # show progress every 5 minute for 5s interval
        fw.flush()
        print("{}".format(round(frame_id/frame_count, 2)), end=" ", flush = True)

      time_label += dt.timedelta(seconds = sample_freq) # advance to next sampling frame
      frame_id += (fps * sample_freq) # FPS x seconds

    print("\n" + "-"*10 + "Total frames processed: ", i, "in", fn)
    process_end = time.time()
    process_time = (process_end - process_start) / 60
    print("-"*10 + "Processing time: ", round(process_time,2), "mins")
    current_video_idx += 1
    
    fw.close()
    cap.release()
  
  # after finishing all the files
  fwlog.close()
  
  _end = time.time()
  total_time = (_end - _start) / 3600
  print("\n"+"="*10 + "Finished all videos, total processing time: ", round(total_time,2), "hrs ")

#-----------------------------------------------------------------
if __name__ == "__main__":

  # global

  #avg_freq = 30 # average every x seconds
  sample_freq = 5 # sample 1 df every 5 seconds

  # define features 
  #label_list = ["max","nonzero_pct","variability", "imax1_ct", "imax2_ct", "imax3_ct", "imax4_ct"]
  label_list = ["max", "nonzero_pct", "variability", "highInt_pct"]
  nonzero_pct_threshold = 0.3
  zero_intensity = 0.1 # threshold for no intensity change
  high_intensity = 3
  save_badframe = False

  # grid_flag = 3 # segmentation regions changed with dataset

  # print("\ngrid_flag = ", grid_flag)
  # if grid_flag == 0: # door, sky, ground, for Radu home
  #   grid_total = 3
  # elif grid_flag == 1: # full + 3x3 grid, for Radu home
  #   grid_total = 10
  # elif grid_flag == 2: # full + sky + building, for AAU only
  #   grid_total = 3
  # elif grid_flag == 3: # for TAMU west campus
  #   grid_total = 6
  # elif grid_flag == -1: # use a single grid of entire frame
  #   grid_total = 1
  # else:
  #   print("Error in grid_flag")

  # profile.runctx('main()', globals(),locals(), sort='cumtime')
  main()

