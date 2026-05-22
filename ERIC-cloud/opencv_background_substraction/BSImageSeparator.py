import time
import os
import sys
import json
import subprocess
import getopt

def copy_images(file_list, path):
	
	print("Copying images to {}......".format(path))
	ct = 0
	for file in file_list:
		cmd = "cp "+file+" "+path
		subprocess.call(cmd, shell=True)		
		ct += 1
		
	print("Copied {} files".format(ct))

def move_images(file_list, img_dir, path):
	
	print("Moving images to {}......".format(path))
	ct = 0
	for file in file_list:
		cmd = "mv "+img_dir+file+" "+path
		subprocess.call(cmd, shell=True)		
		ct += 1
		
	print("Moved {} files".format(ct))

def separate_images(img_list, label_binary_dict, label_intensity_dict, rain_dict):

	rain_day = []
	rain_night = []
	norain_day = []
	norain_night = []

	for img in img_list:
		key, is_day = get_key_from_filename(img)
		if key in rain_dict:
			# print("rain",key)
			label_binary_dict[img] = 1
			label_intensity_dict[img] = rain_dict[key]

			if is_day:
				rain_day.append(img)
			else:
				rain_night.append(img)

		else:
			# print(key)
			label_binary_dict[img] = 0
			label_intensity_dict[img] = 0

			if is_day:
				norain_day.append(img)
			else:
				norain_night.append(img)

	return label_binary_dict, label_intensity_dict, \
				 rain_day, rain_night, \
				 norain_day, norain_night


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

	day_mark = segments[7]
	if int(day_mark):
		is_day = True
	else:
		is_day = False

	# print(key)

	return key, is_day

def main():
	"""
	find all .png pictures under current folder
	separate into rain and norain folder

	Used to prepare images for ResNet model

	$ python BSImageSlicer.py -d test_img/ -f bs_img/ -s bs

	-d: img dir
	-f: output folder
	"""

	t_start = time.time()

	# default
	img_dir = './'
	output_folder = './'
	label_file = './tamu_raindata_rainminute_annotated.json'

	try:
			options, remainder = getopt.getopt(sys.argv[1:], 'd:f:l:s:')
	except getopt.GetoptError as err:
			print("getopt error: %s" % (str(err)))
			return

	for opt, arg in options:
		if opt == '-d':
			img_dir = arg
		if opt == '-f':
			output_folder = arg
		if opt == '-l':
			label_file = arg
		if opt == '-s':
			suffix = arg

	print('img_dir =', img_dir)
	print('output_folder =', output_folder)
	print('label_file =', label_file)
	print('suffix =', suffix)

	# list all .png files
	files = os.listdir(img_dir)

	bs_list = [i for i in files if i.endswith('-{}.png'.format(suffix))]
	print("Total # of *-{}.png: ".format(suffix), len(bs_list))
	print()
	bs_list.sort()

	# load rain label dict
	with open(label_file) as json_file:
		rain_dict = json.load(json_file)

	label_binary_dict = dict()
	label_intensity_dict = dict()

	rgb_rain_day = []
	rgb_rain_night = []
	rgb_norain_day = []
	rgb_norain_night = []

	dif_rain_day = []
	dif_rain_night = []
	dif_norain_day = []
	dif_norain_night = []

	# separate rgb
	label_binary_dict, label_intensity_dict, \
	bs_rain_day, bs_rain_night, \
	bs_norain_day, bs_norain_night = separate_images(bs_list, label_binary_dict, label_intensity_dict, rain_dict)

	print("len(bs_rain_day) =", len(bs_rain_day))
	print("len(bs_rain_night) =", len(bs_rain_night))
	print("len(bs_norain_day) =", len(bs_norain_day))
	print("len(bs_norain_night) =", len(bs_norain_night))
	
	print()

	# move the images to different folders
	folders = ['bs_rain_day', 'bs_rain_night', 'bs_norain_day', 'bs_norain_night']

	for fd in folders:
		path = output_folder + fd

		if not os.path.isdir(path):
			os.mkdir(path)
			print("Folder {} created".format(path))
		else:
			print("Folder {} already existed".format(path))

	# copy the images to corresponding folders
	print()
	move_images(bs_rain_day, img_dir, output_folder+"bs_rain_day")
	move_images(bs_rain_night, img_dir, output_folder+"bs_rain_night")
	move_images(bs_norain_day, img_dir, output_folder+"bs_norain_day")
	move_images(bs_norain_night, img_dir, vim vioutput_folder+"bs_norain_night")

	t_end = time.time()
	print("Total time =", round((t_end - t_start)/60,2), "mins")


if __name__ == "__main__":
	main()