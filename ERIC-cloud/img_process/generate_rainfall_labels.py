#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 12:38:24 2022

@author: tian
"""

import time
import os
import sys
import json
import subprocess
import getopt
import pandas as pd

label_file = '../labels/tamu_raindata_rainminute_annotated.json'

# load rain label dict
with open(label_file) as json_file:
   rain_dict = json.load(json_file)

def pad_two_digits(num):
	res = num
	# print(num)
	# print(res)
	if len(num)==1:
		res = '0'+ num
	# print(res)

	return res

def get_key_from_filename(fn):
	"""
	2021-5-1-15-59-56-53865-1-rgb.png
	"""

	name = fn.split(".")[0]
	segments = name.split("-")
	# print(segments)
	y = segments[0]
	mo = pad_two_digits(segments[1])
	d = pad_two_digits(segments[2])
	h = pad_two_digits(segments[3])
	m = pad_two_digits(segments[4])
	# s = pad_two_digits(segments[5])

	key = y+"-"+mo+"-"+d+" "+h+":"+m

	return key

def get_rainfall_from_name(img):
   
   key = get_key_from_filename(img)
   
   if key in rain_dict:
      rainfall = rain_dict[key]
   else:
      print(key)
      print('warning: images in rain folder not found rainfall')
      rainfall = 0
   
   return rainfall
   

def get_rainfall_label(img_list, subfolder):
   
   folder_list = [subfolder]*len(img_list)
   rainfall_list = []
   
   if subfolder == 'norain': # under norain folder
      rainfall_list = [0]*len(img_list)
      
   else: # under rain folder
      for img in img_list:
         rainfall = get_rainfall_from_name(img)
         rainfall_list.append(rainfall)
   
   res = pd.DataFrame()
   res['subfolder']=folder_list
   res['img_name']=img_list
   res['rainfall(mm/min)']=rainfall_list
   
   return res

def main():
   
   # default
   img_dir = './'
   output_folder = './'
   suffix = 'rgb'
   output_fn = 'rainfall_label_train.csv'
   
   try:
      options, remainder = getopt.getopt(sys.argv[1:], 'd:f:s:o:')
   except getopt.GetoptError as err:
      print("getopt error: %s" % (str(err)))
      return
   
   for opt, arg in options:
      if opt == '-d':
         img_dir = arg
      if opt == '-f':
         output_folder = arg
      if opt == '-s':
         suffix = arg
      if opt == '-o':
         output_fn = arg
   
   print('img_dir =', img_dir)
   print('output_folder =', output_folder)
   print('label_file =', label_file)
   
   # rain folder
   files = os.listdir(img_dir+'rain/')
   img_list = [i for i in files if i.endswith('-{}.png'.format(suffix))]
   print("\nTotal # of -{}.png:".format(suffix), len(img_list))
   img_list.sort()
   
   rainfall_label_rain = get_rainfall_label(img_list, 'rain')
   
   # norain folder
   files = os.listdir(img_dir+'norain/')
   img_list = [i for i in files if i.endswith('-{}.png'.format(suffix))]
   print("\nTotal # of -{}.png:".format(suffix), len(img_list))
   img_list.sort()
   
   rainfall_label_norain = get_rainfall_label(img_list, 'norain')
   
   # concatenate two df
   rainfall_label = pd.concat([rainfall_label_rain, rainfall_label_norain], ignore_index=True)
   
   rainfall_label.sort_values(by=['img_name'], ascending = True)
   # output rainfall_label
   rainfall_label.to_csv('../'+output_fn, index=False)
   print('Complete!')
   
if  __name__ == "__main__":
   main()