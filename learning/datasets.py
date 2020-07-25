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
    
"""
class Yq21Dataset_test(Dataset):
    def __init__(self, path, transforms_=None, transforms2_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.transform2 = transforms.Compose(transforms2_)
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()
        
        self.index = []
        self.location = []
        with open('location.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                index = int(sp_line[0])
                location = int(sp_line[1])
                self.index.append(index)
                self.location.append(location)
        

    def __getitem__(self, index):
        index = index % 10833
        #index = index
        b = [abs(item-index) for item in self.index]
        near_index = b.index(min(b))
        location = self.location[near_index]
        nav = 'S.png'
        if location in [1,17,25,31,37]:
            nav = 'L.png'
        elif location in [11,15,19,21,23,27,33,39,45,49]:
            nav = 'R.png'
        
        #img_path = self.files[index % len(self.files)].strip()
        #img_path = img_path.split(' ')
        
        #path2 = '/home/wang/DataSet/yq_road/'+str(random.randint(0, 30000))+'.jpg'
        #img_A1 = Image.open(path2)
        #img_A1 = img_A1.convert("RGB")
        file_name = str(index)
        file_name = '0'*(10-len(file_name))+file_name
        path_1 = '/mnt/data/public/datasets/YQ/YQ_RAW/2017_03_03/2017_03_03_drive_0005_sync/image_00/data/'+file_name+'.jpg'
        path_2 = '/mnt/data/huifang/YQ_RAW_MASK/2017_03_03/2017_03_03_drive_0005_sync/label_01/'+file_name+'.png'
        
        #img_A1 = Image.open(img_path[0])
        img_A1 = Image.open(path_1)
        #print(img_A1)
        w, h = img_A1.size
        img_A1 = img_A1.crop((0, 0, w, h))

        path = '/home/wang/github/IntersectionClassfication/cGAN/'+nav
        img_A2 = Image.open(path)
        img_A2 = img_A2.convert("RGB")
        #img_A2 = Image.open(img_path[1])
        #w, h = img_A2.size
        #img_A2 = img_A2.crop((0, 0, w, h))
        
        #img_B = Image.open(img_path[2])
        img_B = Image.open(path_2)
        #print(img_B)
        w, h = img_B.size
        img_B = img_B.crop((0, 0, w, h))
        
        img_A1 = self.transform(img_A1)
        img_A2 = self.transform(img_A2)
        img_B = self.transform2(img_B)

        return {'A1': img_A1, 'A2': img_A2, 'B': img_B}

    def __len__(self):
        return 10900#len(self.files)    
"""    
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files = glob.glob(os.path.join(root, mode) + '/*.*')
        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        if mode == 'train':
            self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)


#path = './image_lists/list.txt'
    
class Yq21DatasetLSTM(Dataset):
    def __init__(self, path, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        #f = open(path, 'r')
        #self.files = f.readlines()
        #f.close()

        f = open(path[:-4]+'_id'+path[-4:], 'r')
        self.files_id = f.readlines()
        f.close()

        f = open(path[:-4] + '_all' + path[-4:], 'r')
        self.files_all = f.readlines()
        f.close()

        f = open(path[:-4] + '_id_all' + path[-4:], 'r')
        self.files_all_id = f.readlines()
        f.close()

    def __getitem__(self, index):
        steps = random.choice((4, 5, 6, 7, 8, 9, 10))
        #steps = 4
        #steps = random.choice((10, 12, 14, 16, 18, 20))
        #steps = random.choice((2, 2))
        input_a = []
        input_b = []
        img_id = self.files_id[index % len(self.files_id)].strip()
        img_id = img_id.split(' ')
        img_id = int(img_id[0])

        img_index = self.files_all_id[img_id % len(self.files_all_id)].strip()
        img_index = img_index.split(' ')
        img_index = int(img_index[1])
        randomflag = False
        #if np.random.random() < 0.5:
        #    randomflag = True
        for step in range(0, steps):
            next_id = img_id + step*3
            next_index = self.files_all_id[next_id % len(self.files_all_id)].strip()
            next_index = next_index.split(' ')
            next_index = int(next_index[1])
            if abs(next_index - img_index) > 50:
                continue

            img_path = self.files_all[next_id % len(self.files_all)].strip()

            img_path = img_path.split(' ')
            img_a1 = Image.open(img_path[0])
            img_a2 = Image.open(img_path[1])
            img_b = Image.open(img_path[2])

            # mirror the inputs
            if randomflag:
                img_a1 = Image.fromarray(np.array(img_a1)[:, ::-1, :], 'RGB')
                img_a2 = Image.fromarray(np.array(img_a2)[:, ::-1, :], 'RGB')
                img_b = Image.fromarray(np.array(img_b)[:, ::-1], 'L')

            img_a1 = self.transform(img_a1)
            img_a2 = self.transform(img_a2)
            img_b = self.transform(img_b)

            img_a = torch.cat((img_a1, img_a2), 0)

            input_a.append(img_a)
            input_b.append(img_b)

        return {'A': input_a, 'B': input_b}

    def __len__(self):
        return len(self.files_id)


class Yq21LSTMTest(Dataset):
    def __init__(self, path, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()

    def __getitem__(self, index):
        steps = 4
        input_a1=[]
        input_a2=[]
        input_b = []
        input_f =[]
        for step in range(0, steps):
            id = index + step * 3
            img_path = self.files[id % len(self.files)].strip()
            img_path = img_path.split(' ')

            img_A1 = Image.open(img_path[0])
            w, h = img_A1.size
            img_A1 = img_A1.crop((0, 0, w, h))

            orig_name = img_path[1]
            #new_name = orig_name[:-14] + '/navi_offset/' + orig_name[-14:-4] + '_2.jpg'
            #if step<steps-1:
            img_A2 = Image.open(orig_name)
            #else:
            #    img_A2 = Image.open('/mnt/data/huifang/YQ_RAW_MASK/2017_05_09/2017_05_09_drive_0001_sync/label_01/0000006667.jpg')
            w, h = img_A2.size
            img_A2 = img_A2.crop((0, 0, w, h))


            img_A1 = self.transform(img_A1)
            img_A2 = self.transform(img_A2)

            img_B = Image.open(img_path[2])
            img_B = self.transform(img_B)

            input_a1.append(img_A1)
            input_a2.append(img_A2)
            input_b.append(img_B)
            input_f.append(img_path[0][-14:])

        return {'A1': input_a1, 'A2': input_a2, 'B': input_b, 'F': input_f}

    def __len__(self):
        return len(self.files)

class Yq21DatasetE2E(Dataset):
    def __init__(self, path, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        f = open(path[:-4]+'_id'+path[-4:], 'r')
        self.files_id = f.readlines()
        f.close()

        f = open(path[:-4] + '_all' + path[-4:], 'r')
        self.files_all = f.readlines()
        f.close()

        f = open(path[:-4] + '_id_all' + path[-4:], 'r')
        self.files_all_id = f.readlines()
        f.close()

    def __getitem__(self, index):
        steps = 4
        #steps = random.choice((10, 12, 14, 16, 18, 20))
        #steps = random.choice((2, 2))
        input_a1 = []
        input_a2 = []
        input_b = []
        img_id = self.files_id[index % len(self.files_id)].strip()
        img_id = img_id.split(' ')
        img_id = int(img_id[0])

        img_index = self.files_all_id[img_id % len(self.files_all_id)].strip()
        img_index = img_index.split(' ')
        img_index = int(img_index[1])

        cropcnt = round(np.random.random() * 20)
        for step in range(0, steps):
            next_id = img_id + step*3
            next_index = self.files_all_id[next_id % len(self.files_all_id)].strip()
            next_index = next_index.split(' ')
            next_index = int(next_index[1])
            if abs(next_index - img_index) > 50:
                continue

            img_path = self.files_all[next_id % len(self.files_all)].strip()

            img_path = img_path.split(' ')
            img = Image.open(img_path[0])
            navi = Image.open(img_path[1])
            v = eval(img_path[2])
            w = eval(img_path[3])

            # crop current perception
            # img_in: 314 648
            # img_out: 314 518
            keepsize = 628
            img = img.crop([cropcnt, 0, cropcnt + keepsize, img.size[1]])

            img = self.transform(img)
            navi = self.transform(navi)

            input_a1.append(img)
            input_a2.append(navi)
            input_b.append([v, w])

        return {'A1': input_a1, 'A2': input_a2[-1], 'B': input_b[-1]}

    def __len__(self):
        return len(self.files_id)


class Yq21DatasetE2E_test(Dataset):
    def __init__(self, path, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')

        img_a1 = Image.open(img_path[0])
        w, h = img_a1.size
        img_a1 = img_a1.crop((0, 0, w, h))

        img_a2 = Image.open(img_path[1])
        w, h = img_a2.size
        img_a2 = img_a2.crop((0, 0, w, h))

        v = eval(img_path[2])
        w = eval(img_path[3])

        img_a1 = self.transform(img_a1)
        img_a2 = self.transform(img_a2)
        motion = [v, w]
        return {'A1': img_a1, 'A2': img_a2, 'B': motion}

    def __len__(self):
        return len(self.files)