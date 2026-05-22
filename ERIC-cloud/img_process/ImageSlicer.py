# -*- coding: utf-8 -*-
"""
"""

import cv2
import matplotlib.pyplot as plt
import datetime as dt
import os.path
import os
import time
import getopt
import sys

#------------------------------------------------------------------------------
def plot_img(img):
  fig, axs = plt.subplots(1,1,figsize=(8,8))
  axs.imshow(img[:,:,::-1])
  plt.show()
  plt.close()

#------------------------------------------------------------------------------
def plot_grayimg(grayimg, fn, output_path = None):

  # fig, axs = plt.subplots(1,1,figsize=(1.536,2.048))

  # im = axs.imshow(grayimg, cmap='gray', vmin=0, vmax=3, norm = None)
  # im.axes.get_xaxis().set_visible(False)
  # im.axes.get_yaxis().set_visible(False)
  # plt.savefig(output_path+fn, dpi=1000, bbox_inches='tight', pad_inches = 0)

  # fig.colorbar(im, ax=axs)

  fig = plt.figure(figsize=(1.536, 2.048), frameon=False, dpi=1000)
  # figure = plt.gcf()
  # figure.set_size_inches(1.536, 2.048)

  im = plt.imshow(grayimg, cmap='gray', vmin=0, vmax=3)
  plt.axis('off')
  im.axes.get_xaxis().set_visible(False)
  im.axes.get_yaxis().set_visible(False)
  plt.savefig(output_path+fn, dpi=1000)
  # plt.savefig(output_path+fn, dpi=1000, bbox_inches='tight', pad_inches = 0)

  plt.close()

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
    # fn_m = fn[:17] + "00.mp4" # fix the start second to 00
    time = dt.datetime.strptime(fn,"%Y-%m-%dT%H-%M-%S.mp4")

  else:
    time = dt.datetime.strptime(fn[:24],"doorbell-%Y%m%d-%H%M%S")

  t = str(time.year)+"-"+str(time.month)+"-"+str(time.day)+" "+str(time.hour)+":"+str(time.minute)+":"+str(time.second)

  print("video starting time: ",t)

  return time

def get_time_string(time):

  time_str = str(time.year)+"-"+str(time.month)+"-"+str(time.day)+"-"+str(time.hour)+"-"+str(time.minute)+"-"+str(time.second)

  return time_str

# def get_rain(time_label, rain_dict):

#   key = 
#   if rain_dict[key] == 0:
#     is_rain = 0
#   else:
#     is_rain = 1

#   return is_rain

#------------------------------------------------------------------------------
def main():
  """
  This script slice images from video files and saved in local folders

  Used to prepare images for ResNet model

  $ python process_video_features.py -i 2021-08-15/

  -i: video_dir
  -o: image_dir
  -s: video_start (inclusive, from 1)
  -e: video_end (inclusive, from 1)
  """

  _start = time.time()

  # default
  video_dir = './'
  image_dir = './Images/'
  slice_rgb = False
  slice_dif = False

  try:
      options, remainder = getopt.getopt(sys.argv[1:], 'i:o:s:e:f:')
  except getopt.GetoptError as err:
      print("getopt error: %s" % (str(err)))
      return

  for opt, arg in options:
    if opt == '-i':
      video_dir = arg
    if opt == '-o':
      image_dir = arg
    if opt == '-f':
      image_format = arg

  if image_format == 'rgb':
    slice_rgb = True
  elif image_format == 'dif':
    slice_dif = True
  elif image_format == 'rgbdif':
    slice_rgb = True
    slice_dif = True

  print('video_dir =', video_dir)
  print('image_dir =', image_dir)

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
  print('video_end (inclusive) = ', video_end)

  current_video_idx = video_start

  # create the image folder
  if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
    print("Folder {} created".format(image_dir))
  else:
    print("Folder {} already existed".format(image_dir))

  for fn in fn_list[video_start-1 : video_end]: # loop through each video file

    print()
    print("-"*10 + "Slicing video file:", str(current_video_idx) + '/' + str(len(fn_list)), fn)
    process_start = time.time()

    time_label = get_time_from_filename(fn)
    th = time_label.hour
    if th < 7 or th >= 20: # light turns off at 7am, turns on at 8pm
      is_day = 0
    else:
      is_day = 1

    # Read the video file info
    cap = cv2.VideoCapture(video_dir + fn)
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
    frame_id = 15 # FPS = 30, take middle point

    while(cap.isOpened() and frame_id <= frame_count) : # loop through the frames
  
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id-1)
      success, frame1 = cap.read()
      if not success:
        print("error: fail to read the frame", frame_id-1)
        break
      else: 
        i += 1

      cur_time = get_time_string(time_label)

      if slice_rgb:
        # save current frame image
        img_name = cur_time + "-" + str(frame_id) + "-" + str(is_day) + "-rgb.png"
        cv2.imwrite(image_dir+img_name, frame1)

      if slice_dif:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame2 = cap.read()
        if not success:
          print("error: fail to read the frame", frame_id)
          break
        else: 
          i += 1

        gray_f1 = frame1.mean(axis=2)
        gray_f2 = frame2.mean(axis=2)
        gray_df = abs(gray_f2 - gray_f1)

        # crop out the caption
        # plot_img(frame1[1776:1811,480:1050,:])
        gray_df[1776:1811,480:1050] = 0
       
        # get rain label
        # is_rain = get_rain(time_label, rain_dict)
        # grayimg_name = cur_time + "-" + str(frame_id) + "-" + str(is_day) + "-" +str(is_rain) + "-dif.png"

        grayimg_name = cur_time + "-" + str(frame_id) + "-" + str(is_day) + "-dif.png"
        # save the rain streaks image
        plot_grayimg(gray_df, grayimg_name, image_dir)

        # cv2.imwrite(image_dir+grayimg_name, gray_df) # pure dark, no seen feature
      
      if i%120 == 0:
          print(round(frame_id/frame_count,2), end=' ')

      time_label += dt.timedelta(seconds = sample_freq) # advance to next sampling frame
      frame_id += (fps * sample_freq) # FPS x seconds

    print()
    print("-"*10 + "Total frames processed: ", i, "in", fn)
    process_end = time.time()
    process_time = (process_end - process_start) / 60
    print("-"*10 + "Processing time: ", round(process_time,2), "mins")
    current_video_idx += 1
    
    cap.release()
  
  # after finishing all the files
  _end = time.time()
  total_time = (_end - _start) / 3600
  print("\n"+"="*10 + "Finished slicing all videos, total processing time: ", round(total_time,2), "hrs ")

  return 1

#-----------------------------------------------------------------
if __name__ == "__main__":

  # global
  sample_freq = 5 # sample 1 pic every 5 seconds
  grid_flag = 3 # tamu west campus

  main()

