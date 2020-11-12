import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
    
class Yq21Dataset(Dataset):
    def __init__(self, transforms_=None, transforms2_=None,data_index=[1,2,3,4,5]):
        self.transform = transforms.Compose(transforms_)
        self.transform2 = transforms.Compose(transforms2_)
        self.data_index = data_index
        self.files_dict = {}
        self.total_len = 0
        for index in self.data_index:
            files = glob.glob('/media/wang/DATASET/images'+str(index)+'/*.png')
            self.total_len += len(files)
            file_names = []
            for file in files:
                file_name = file.split('/')[5]
                file_names.append(file_name)
            file_names.sort()
            self.files_dict[index] = file_names
        
        self.corner_start = [
                [250,2750,3750,4400,5050,5750,6550,7450,8150,8650,9350,10250,11150,12450],#1
                [142,2492,3442,3942,4392,4942,5342,6042,6292,6942,7692,8042,8689,9539,10439,11639],#2
                [286,2536,3386,3886,4336,4886,5286,5836,6086,6686,7336,7786,8336,9186,9986,11186],#3
                [324,2524,3374,3924,4274,5224,5824,6124,6724,7424,7924,8424,9324,10174,11374],#4
                [149,2399,3199,3699,4149,4749,5199,5649,6599,7299,7749,8299,9099,9899,11149],#5
                [293,2593,3443,3943,4343,4893,5193,5793,6693,7393,7693,8343,9193,9993,11093],#6
                ]
        self.corner_end = [
                [750,3350,4200,4750,5150,6150,7150,7850,8550,8950,9550,10650,11450,12750],#1
                [442,2742,3642,4192,4542,5142,5642,6242,6542,7242,7892,8339,8889,9839,10689,11939],#2
                [536,2836,3636,4136,4536,5086,5486,5986,6336,6986,7586,8086,8586,9486,10236,11436],#3
                [624,2824,3674,4124,4424,5424,6024,6324,7074,7724,8174,8724,9624,10424,11724],#4
                [449,2599,3499,3999,4349,4899,5399,6299,6949,7549,7949,8499,9399,10199,11449],#5
                [593,2843,3693,4193,4543,5143,5393,6393,6993,7593,7993,8593,9443,10243,11493],#6
                ]

    def __getitem__(self, index):
        mirror = True if random.random() > 0.5 else False
        data_index = random.sample(self.data_index,1)[0]
        
        corner_start = self.corner_start[data_index-1]
        corner_end = self.corner_end[data_index-1]
        try:
            if random.random() > 0.5:
                corner_index = random.sample([item for item in range(len(corner_start))],1)[0]
                file_index = random.sample([item for item in range(corner_start[corner_index], corner_end[corner_index])],1)[0]
                file_name = self.files_dict[data_index][file_index]
            else:
                file_name = random.sample(self.files_dict[data_index],1)[0]
        except:
            print('Error when get corner data')
            file_name = random.sample(self.files_dict[data_index],1)[0]
        
        #file_name = random.sample(self.files_dict[data_index],1)[0]
        
        image_path = '/media/wang/DATASET/images'+str(data_index)+'/'+file_name
        label_path = '/media/wang/DATASET/label'+str(data_index)+'/'+file_name
        nav_path = '/media/wang/DATASET/nav'+str(data_index)+'/'+file_name

        img = Image.open(image_path).convert("RGB")
        nav = Image.open(nav_path).convert("RGB")
        lable = Image.open(label_path).convert('L')

        # mirror the inputs
        if mirror:
            img = Image.fromarray(np.array(img)[:, ::-1, :], 'RGB')
            nav = Image.fromarray(np.array(nav)[:, ::-1, :], 'RGB')
            lable = Image.fromarray(np.array(lable)[:, ::-1], 'L')

        img = self.transform(img)
        nav = self.transform(nav)
        lable = self.transform2(lable)
        input_img = torch.cat((img, nav), 0)

        return {'A': input_img, 'B': lable}

    def __len__(self):
        return self.total_len

class Yq21Dataset_test(Dataset):
    def __init__(self, transforms_=None, transforms2_=None,data_index=[6]):
        self.transform = transforms.Compose(transforms_)
        self.transform2 = transforms.Compose(transforms2_)
        self.data_index = data_index
        self.files_dict = {}
        self.total_len = 0
        for index in self.data_index:
            files = glob.glob('/media/wang/DATASET/images'+str(index)+'/*.png')
            self.total_len += len(files)
            file_names = []
            for file in files:
                file_name = file.split('/')[5]
                file_names.append(file_name)
            file_names.sort()
            self.files_dict[index] = file_names
            
        self.corner_start = [
                [250,2750,3750,4400,5050,5750,6550,7450,8150,8650,9350,10250,11150,12450],#1
                [142,2492,3442,3942,4392,4942,5342,6042,6292,6942,7692,8042,8689,9539,10439,11639],#2
                [286,2536,3386,3886,4336,4886,5286,5836,6086,6686,7336,7786,8336,9186,9986,11186],#3
                [324,2524,3374,3924,4274,5224,5824,6124,6724,7424,7924,8424,9324,10174,11374],#4
                [149,2399,3199,3699,4149,4749,5199,5649,6599,7299,7749,8299,9099,9899,11149],#5
                [293,2593,3443,3943,4343,4893,5193,5793,6693,7393,7693,8343,9193,9993,11093],#6
                ]
        self.corner_end = [
                [750,3350,4200,4750,5150,6150,7150,7850,8550,8950,9550,10650,11450,12750],#1
                [442,2742,3642,4192,4542,5142,5642,6242,6542,7242,7892,8339,8889,9839,10689,11939],#2
                [536,2836,3636,4136,4536,5086,5486,5986,6336,6986,7586,8086,8586,9486,10236,11436],#3
                [624,2824,3674,4124,4424,5424,6024,6324,7074,7724,8174,8724,9624,10424,11724],#4
                [449,2599,3499,3999,4349,4899,5399,6299,6949,7549,7949,8499,9399,10199,11449],#5
                [593,2843,3693,4193,4543,5143,5393,6393,6993,7593,7993,8593,9443,10243,11493],#6
                ]

    def __getitem__(self, index):
        data_index = random.sample(self.data_index,1)[0]
        #file_name = random.sample(self.files_dict[data_index],1)[0]
        
        corner_start = self.corner_start[data_index-1]
        corner_end = self.corner_end[data_index-1]
        try:
            if random.random() > 0.5:
                corner_index = random.sample([item for item in range(len(corner_start))],1)[0]
                file_index = random.sample([item for item in range(corner_start[corner_index], corner_end[corner_index])],1)[0]
                file_name = self.files_dict[data_index][file_index]
            else:
                file_name = random.sample(self.files_dict[data_index],1)[0]
        except:
            print('Error when get corner data')
            file_name = random.sample(self.files_dict[data_index],1)[0]
            
        image_path = '/media/wang/DATASET/images'+str(data_index)+'/'+file_name
        label_path = '/media/wang/DATASET/label'+str(data_index)+'/'+file_name
        nav_path = '/media/wang/DATASET/nav'+str(data_index)+'/'+file_name

        img = Image.open(image_path).convert("RGB")
        nav = Image.open(nav_path).convert("RGB")
        lable = Image.open(label_path).convert('L')

        img = self.transform(img)
        nav = self.transform(nav)
        lable = self.transform2(lable)

        return {'A1': img, 'A2': nav, 'B': lable}

    def __len__(self):
        return self.total_len
    
    
class Yq21Dataset_eval(Dataset):
    def __init__(self, transforms_=None, transforms2_=None,data_index=[6]):
        self.transform = transforms.Compose(transforms_)
        self.transform2 = transforms.Compose(transforms2_)
        self.data_index = data_index
        self.files_dict = {}
        self.total_len = 0
        for index in self.data_index:
            files = glob.glob('/media/wang/DATASET/images'+str(index)+'/*.png')
            self.total_len += len(files)
            file_names = []
            for file in files:
                file_name = file.split('/')[5]
                file_names.append(file_name)
            file_names.sort()
            self.files_dict[index] = file_names

        self.cnt = 0
            

    def __getitem__(self, index):
        data_index = random.sample(self.data_index,1)[0]
        file_name = self.files_dict[data_index][self.cnt]
        self.cnt += 1
            
        image_path = '/media/wang/DATASET/images'+str(data_index)+'/'+file_name
        label_path = '/media/wang/DATASET/label'+str(data_index)+'/'+file_name
        nav_path = '/media/wang/DATASET/nav'+str(data_index)+'/'+file_name

        img = Image.open(image_path).convert("RGB")
        nav = Image.open(nav_path).convert("RGB")
        lable = Image.open(label_path).convert('L')

        img = self.transform(img)
        nav = self.transform(nav)
        lable = self.transform2(lable)

        return {'A1': img, 'A2': nav, 'B': lable, 'file_name':file_name}

    def __len__(self):
        return self.total_len