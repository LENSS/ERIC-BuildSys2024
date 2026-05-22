import time
import os
import sys
import json
import subprocess
import getopt

def move_images(file_list, img_dir, path):
	
	print("Moving images to {}......".format(path))
	ct = 0
	for file in file_list:
		cmd = "mv "+img_dir+file+" "+path
		subprocess.call(cmd, shell=True)		
		ct += 1
		
	print("Copied {} files".format(ct))

def separate_images(img_list, rain_dict):

	rain = []
	norain = []

	for img in img_list:
		key = get_key_from_filename(img)
		if key in rain_dict:
			rain.append(img)
		else:
			norain.append(img)

	return rain, norain

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

def main():
	"""
	find all .png pictures under current folder
	separate into rain and norain folder

	Used to prepare images for ResNet model

	-d: img dir
	-f: output folder
	"""

	t_start = time.time()

	# default
	img_dir = './'
	output_folder = './'
	label_file = '../labels/tamu_raindata_rainminute_annotated.json'
	suffix = 'rgb'

	try:
		options, remainder = getopt.getopt(sys.argv[1:], 'd:f:s:')
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

	print('img_dir =', img_dir)
	print('output_folder =', output_folder)
	print('label_file =', label_file)
	print('suffix =', suffix)

	# list all -suffix.png files
	files = os.listdir(img_dir)
	img_list = [i for i in files if i.endswith('-{}.png'.format(suffix))]

	print("\nTotal # of -{}.png:".format(suffix), len(img_list))
	img_list.sort()

	# load rain label dict
	with open(label_file) as json_file:
		rain_dict = json.load(json_file)

	rain, norain = separate_images(img_list, rain_dict)

	print("len(rain) =", len(rain))
	print("len(norain) =", len(norain))
	print()

	# move the images to different folders
	folders = ['','rain', 'norain']

	for fd in folders:
		path = output_folder+fd
		if not os.path.isdir(path):
			os.mkdir(path)
			print("Folder {} created".format(path))
		else:
			print("Folder {} already existed".format(path))

	# copy the images to corresponding folders
	print()
	move_images(rain, img_dir, output_folder+"rain")
	move_images(norain, img_dir, output_folder+"norain")

	t_end = time.time()
	print("Total time =", round((t_end - t_start)/60,2), "mins")


if __name__ == "__main__":
	main()