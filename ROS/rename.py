import os
import glob

def read_pcd():
    files = glob.glob('/home/wang/github/RoBoCar/ROS/pcd/*.pcd')
    file_path = []
    for file in files:
        ts = file.split('/')[7][:-4]
        file_path.append(ts)
        
    file_path.sort()
    return file_path
    
    
file_path = read_pcd()

for i in range(len(file_path)):
    file_name = file_path[i]
    src = '/home/wang/github/RoBoCar/ROS/pcd/'+file_name+'.pcd'
    dst = '/home/wang/github/RoBoCar/ROS/out/'+str(i)+'.pcd'
    os.rename(src, dst)