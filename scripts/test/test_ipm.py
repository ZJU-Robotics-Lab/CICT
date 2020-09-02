import cv2
import glob
import random
import copy
import numpy as np
from PIL import Image

dataset_path = '/media/wang/DATASET/CARLA_HUMAN/town01/1/'

files = glob.glob(dataset_path+'/pm/*.png')
file_names = []
for file in files:
    file_name = file.split('/')[-1][:-4]
    file_names.append(file_name)
file_names.sort()

def get_one():
	file_name = random.sample(file_names, 1)[0]

	img_path = dataset_path +'/img/'+file_name+'.png'
	img = Image.open(img_path).convert("RGB")
	                
	label_path = dataset_path + '/pm/'+file_name+'.png'
	label = Image.open(label_path).convert('L')

	_img = np.asarray(img)
	img = copy.deepcopy(_img)
	img.flags.writeable = True
	label = np.asarray(label)
	print(img.shape, label.shape)


	mask = np.where(label > 200)
	image_uv = np.stack([mask[1],mask[0]])

	img[mask[0],mask[1]] = (255, 0, 0)

	img = Image.fromarray(img, 'RGB')
	img.show()

for _ in range(10):
	get_one()