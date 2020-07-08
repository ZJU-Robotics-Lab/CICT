import math
from PIL import Image
import matplotlib.pyplot as plt
from manual_gps import dist_p2p,manual_gps_y, manual_gps_x,find_nn,gen_manual_gps
from tqdm import tqdm

def avg(values):
    summ = sum(values)
    return summ/len(values)
  
def normal(values, is_y=False):
    avg_value = values[0]
    if is_y:
        avg_value = (11589596.333751025+11589605.59585501)/2
    else:
        avg_value = (3047362.377753752+3047362.981469983)/2
    return [(item - avg_value) for item in values]
   

def filt_gps(x_list, y_list, ts_list):
    filt_x = [-x_list[0]*scale_x + x_offset]
    filt_y = [y_list[0]*scale_y + y_offset]
    last_ts = float(ts_list[0])
    for i in range(len(x_list)-1):
        #ts = float(ts_list[i])
        next_s = float(ts_list[i+1])
        dt = next_s - last_ts
        dist = dist_p2p(x_list[i], y_list[i], x_list[i+1], y_list[i+1])
        v = dist/dt

        if v > 0.0:
            last_ts = next_s
            
        if v < 2 and v > 0.0:
            x = x_list[i]
            y = y_list[i]
            _x = -x*scale_x + x_offset
            _y = y*scale_y + y_offset
            filt_x.append(_x)
            filt_y.append(_y)
        else:
            filt_x.append(filt_x[-1])
            filt_y.append(filt_y[-1])
            
    return filt_x, filt_y

data_index = 4
scale_x = 6.0
scale_y = 6.0
x_offset = 3300
y_offset = 2600
        
xs = []
ys = []
ts = []
with open('/media/wang/DATASET/gps'+str(data_index)+'/gps.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        try:
            sp_line = line.split()
            timestamp = sp_line[0]
            data = sp_line[2][1:-3].split('\\t')
            x = float(data[0])
            y = float(data[1])
            xs.append(x)
            ys.append(y)
            ts.append(timestamp)
        except:
            pass
        
nx = normal(xs)
ny = normal(ys, True)
fter_x, fter_y = filt_gps(nx, ny, ts)
#get_locations(data_index)
#img_ts_list, img_icp_ts_index = read_imgs(data_index)

gps_x, gps_y = gen_manual_gps(manual_gps_x, manual_gps_y)

start = 0
end = len(fter_y)
img = Image.open('map-mark.png')
fig,ax = plt.subplots(figsize=(15, 15))

plt.plot(fter_y[start:end],fter_x[start:end],'xkcd:red',linewidth=4)
plt.plot(gps_y,gps_x,'xkcd:green',linewidth=4)
plt.axis('off')
plt.tight_layout()

def get_nav(img, angle, i, index):
    size_y = 800
    size_x = 400

    img2 = img.crop((gps_y[index]-size_y, gps_x[index]-size_x, gps_y[index]+size_y, gps_x[index]+size_x))
    im_rotate = img2.rotate(angle)
    size_y2 = 100#300
    size_x2 = 50#150
    img3 = im_rotate.crop((img2.size[0]//2-size_y2, img2.size[1]//2-size_x2, img2.size[0]//2+size_y2, img2.size[1]//2+size_x2))
    #ax.imshow(img3)
    img3.save('/media/wang/DATASET/nav'+str(data_index)+'/'+ts[i]+'.png')


dy = (fter_y[6] - fter_y[3]) + (fter_y[5] - fter_y[2]) + (fter_y[4] - fter_y[1]) + (fter_y[3] - fter_y[0])
dx = (fter_x[6] - fter_x[3]) + (fter_x[5] - fter_x[2]) + (fter_x[4] - fter_x[1]) + (fter_x[3] - fter_x[0])
last_angle = 180.*math.atan2(dy, dx)/math.pi

def angle_normal(angle):
    while angle < -180. or angle > 180.:
        if angle < -180.:
            angle += 180.
        elif angle > 180.:
            angle -= 180.
    return angle

for i in tqdm(range(len(fter_x))):
    nn_x, nn_y, nn_index = find_nn(fter_x[i], fter_y[i], gps_x, gps_y)
    dy = (gps_y[min(nn_index+5,len(fter_x)-1)] - gps_y[max(0,nn_index-5)])
    dx = (gps_x[min(nn_index+5,len(fter_x)-1)] - gps_x[max(0,nn_index-5)])
    angle = 180.*math.atan2(dy, dx)/math.pi

    input_angle = 0.5*last_angle+0.5*angle
    last_angle = input_angle
    
    get_nav(img, -input_angle+180., i, nn_index)