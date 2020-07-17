import cv2
import numpy as np

pix_width = 0.05
map_x_min = 0.0
map_x_max = 10.0
map_y_min = -10.0
map_y_max = 10.0
lim_z = -1.5

width = int((map_x_max-map_x_min)/pix_width)
height = int((map_y_max-map_y_min)/pix_width)
    
u_bias = int(np.abs(map_y_max)/(map_y_max-map_y_min)*height)
v_bias = int(np.abs(map_x_max)/(map_x_max-map_x_min)*width)

def read_pcd(file_path):
    x = []
    y = []
    z = []
    intensity = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        [lines.pop(0) for _ in range(11)]
        for line in lines:
            sp_line = line.split()
            if float(sp_line[0]) < 0:
                continue
            x.append(float(sp_line[0]))
            y.append(float(sp_line[1]))
            z.append(float(sp_line[2]))
            intensity.append(float(sp_line[3]))
    return np.array([x, y, z]), intensity

def project(x, y):
    u = -x/pix_width + u_bias
    v = -y/pix_width + v_bias
    result = np.array([u, v])
    mask = np.where((result[0] < width) & (result[1] < height))
    result = result[:, mask[0]]
    return result.astype(np.int16)
    
def get_cost_map(trans_pc, point_cloud, show=False):
    img = np.zeros((width,height,1), np.uint8)
    img.fill(0)
    img2 = np.zeros((width,height,1), np.uint8)
    img2.fill(255)
    
    res = np.where((trans_pc[0] > map_x_min) & (trans_pc[0] < map_x_max) & (trans_pc[1] > map_y_min) & (trans_pc[1] < map_y_max)) 
    trans_pc = trans_pc[:, res[0]]
    u, v = project(trans_pc[0], trans_pc[1])
    img[u,v]=255
    
    #res = np.where((point_cloud[0] > map_x_min) & (point_cloud[0] < map_x_max) & (point_cloud[1] > map_y_min) & (point_cloud[1] < map_y_max)) 
    res = np.where((point_cloud[2] > lim_z) & (point_cloud[0] > map_x_min) & (point_cloud[0] < map_x_max) & (point_cloud[1] > map_y_min) & (point_cloud[1] < map_y_max)) 
    point_cloud = point_cloud[:, res[0]]
    u, v = project(point_cloud[0], point_cloud[1])
    img2[u,v] = 0
    
    kernel = np.ones((25,25),np.uint8)  
    img2 = cv2.erode(img2,kernel,iterations = 1)
    
    img = cv2.addWeighted(img,0.5,img2,0.5,0)
    kernel_size = (21, 21)
    sigma = 25
    img = cv2.GaussianBlur(img, kernel_size, sigma);
    if show:
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


m = 50#angles number
max_theta = 2*np.pi/3
L = 8.0# path length
Length = 1.448555
vel = 0.5
n = 100# points number

collision_penalty = 1.0
half_width = 10
collision_threshhold = 70

def gen_r():
    rs = []
    for i in range(m):
        theta = i*2.0*max_theta/m - max_theta
        if np.abs(theta - 0.) < 0.00001:
            rs.append(9999999999999)
            continue
        r = L/theta
        rs.append(r)
        rs.append(-r)
    return rs

def get_cmd(img, show=False, save=False, file_name=None):
    rs = gen_r()
    for i in range(m):
        theta = i*2.0*max_theta/m - max_theta
        if np.abs(theta - 0.) < 0.00001:
            rs.append(99999)
            continue
        r = L/theta
        rs.append(r)
        rs.append(-r)
    
    best_cost = 0
    best_r = 99999
    best_u = []
    best_v = []
    
    for r in rs:
        theta = L/np.abs(r)
        cost = 0
        indexs = np.arange(n)
        xs = np.abs(r*np.sin(indexs*theta/n))
        ys = r*(1-np.cos(indexs*theta/n))
        u, v = project(xs, ys)
        v2 = np.clip(v+half_width, 0, height-1)
        v3 = np.clip(v-half_width, 0, height-1)
        
        mask = np.where(img[u,v] < collision_threshhold)[0]
        mask2 = np.where(img[u,v2] < collision_threshhold)[0]
        mask3 = np.where(img[u,v3] < collision_threshhold)[0]
        all_collision = len(mask)+len(mask2)+len(mask3)
        
        cost = sum(img[u,v]/255.0)+sum(img[u,v2]/255.0)+sum(img[u,v3]/255.0)-collision_penalty*all_collision
        #img[u, v] = 0
        
        if best_cost < cost:
            best_cost = cost
            best_r = r
            best_u = u
            best_v = v

    img[best_u,best_v] = 0

    if show:
        cv2.imshow('Result', img)
        cv2.waitKey(100)
        #cv2.destroyAllWindows()

    direct = -1.0 if best_r > 0 else 1.0
    yaw = direct*np.arctan2(Length, abs(best_r))
    return yaw