import math
import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


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
   

class Filter():
    def __init__(self):
        self.v = 0.0
        self.yaw = 0.0

        self.x = 0.0
        self.y = 0.0
        self.last_t = time.time()
        self.cnt = 0
        self.INIT_STEPS = 50
        self.ALPHA = 0.2
        self.MAX_V = 50.0
        
        self.x_his = []
        self.y_his = []
        
        self.x_bias = 0.0
        self.y_bias = 0.0
        
    
    def dist(self, x, y):
        return math.sqrt((self.x-x)**2 + (self.y-y)**2)
        
    def step(self, x, y):
        """
        if self.cnt >= self.INIT_STEPS:
            x = x - self.x_bias
            y = y - self.y_bias
        """
        self.cnt += 1
        now = time.time()

        d = self.dist(x, y)
        dt = 0.2#now - self.last_t
        v = d/dt
        
        if self.cnt < self.INIT_STEPS-10:
            self.x = (1-self.ALPHA)*self.x + self.ALPHA*x
            self.y = (1-self.ALPHA)*self.y + self.ALPHA*y
            self.x_bias = self.x
            self.y_bias = self.y
            
        elif self.cnt < self.INIT_STEPS:
            self.x = (1-self.ALPHA)*self.x + self.ALPHA*x
            self.y = (1-self.ALPHA)*self.y + self.ALPHA*y
            self.x_his.append(x)
            self.y_his.append(y)
        else:
            if v > self.MAX_V:
                self.x = self.x + math.cos(self.yaw)*self.v*dt
                self.y = self.y + math.sin(self.yaw)*self.v*dt
            else:
                self.v = (1-self.ALPHA)*self.v + self.ALPHA*v
                self.x = x
                self.y = y
                
                self.yaw = math.atan2((y-self.y_his[5]), (x-self.y_his[5]))
        
            self.x_his.pop(0)
            self.y_his.pop(0)
            self.x_his.append(self.x)
            self.y_his.append(self.y)
            
        self.last_t = now
        return self.x, self.y, self.v


scale_x = 6.0
scale_y = 6.0
x_offset = 3300
y_offset = 2600
        
xs = []
ys = []
ts = []
with open('./gps2/gps.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        try:
            sp_line = line.split()
            timestamp = float(sp_line[0])
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

fter = Filter()
fter_x = []
fter_y = []
fter_v = []

manual_gps_y = [2400, 2300, 3815, 3630, 3325, 3240, 2855, 2570, 2420, 2465, 2115, 2260, 2095, 2075, 2200, 1655, 1820, 2605, 2460]
manual_gps_x = [3200, 3430, 4000, 4520, 4400, 4600, 4665, 4555, 4240, 4090, 3970, 3555, 3490, 3395, 3055, 2820, 2335, 2615, 3045]

def find_nn(x, y):
    gps_y = []
    gps_x = []
    


for i in range(len(nx)):
    x, y, v = fter.step(nx[i], ny[i])
    _x = -x*scale_x  + x_offset
    _y = y*scale_y  + y_offset
    fter_x.append(_x)
    fter_y.append(_y)
    fter_v.append(v)

start = 1000#4200#700
end = -3000#-1400#-3000
img = Image.open('map-mark.png')
fig,ax = plt.subplots(figsize=(15, 15))
#plt.plot(fter_y[start:end],fter_x[start:end],'xkcd:red',linewidth=4)
plt.plot(manual_gps_y,manual_gps_x,'xkcd:blue',linewidth=5)
plt.axis('off')
plt.tight_layout()
#img2 = img.crop((2390-800, 3270-1000, 2390+1500, 3270+1500))
ax.imshow(img)
#fig.savefig('out5.pdf', bbox_inches='tight', dpi=1000)
"""
for i in range(start, len(fter_x)+end-15):
    draw = ImageDraw.Draw(img)
    draw.line((fter_y[i], fter_x[i], fter_y[i+2], fter_x[i+2]), 'red', width=1)

ax.imshow(img)
fig.savefig('out6.pdf', bbox_inches='tight', dpi=1000)
"""
def get_nav(img, angle, index=0):
    size_y = 800
    size_x = 400
    r=10
    img2 = img.crop((fter_y[index]-size_y, fter_x[index]-size_x, fter_y[index]+size_y, fter_x[index]+size_x))
    #drawObject = ImageDraw.Draw(img2)
    #drawObject.ellipse((img2.size[0]//2,img2.size[1]//2,img2.size[0]//2+r,img2.size[1]//2+r),fill="red")
    im_rotate = img2.rotate(angle)
    size_y2 = 100#300
    size_x2 = 50#150
    img3 = im_rotate.crop((img2.size[0]//2-size_y2, img2.size[1]//2-size_x2, img2.size[0]//2+size_y2, img2.size[1]//2+size_x2))
    #ax.imshow(img3)
    img3.save('output2/'+str(ts[index])+'.png')

"""
i=start
dy = (fter_y[i+6] - fter_y[i+3]) + (fter_y[i+5] - fter_y[i+2]) + (fter_y[i+4] - fter_y[i+1]) + (fter_y[i+3] - fter_y[i])
dx = (fter_x[i+6] - fter_x[i+3]) + (fter_x[i+5] - fter_x[i+2]) + (fter_x[i+4] - fter_x[i+1]) + (fter_x[i+3] - fter_x[i])
last_angle = 180.*math.atan2(dy, dx)/math.pi

def angle_normal(angle):
    while angle < -180. or angle > 180.:
        if angle < -180.:
            angle += 180.
        elif angle > 180.:
            angle -= 180.
    return angle

total = len(fter_x)+end-6-start
for i in range(start, len(fter_x)+end-6):
    dy = (fter_y[i+4] - fter_y[i+0]) + (fter_y[i+3] - fter_y[i-1]) + (fter_y[i+2] - fter_y[i-2]) + (fter_y[i+1] - fter_y[i-3])
    dx = (fter_x[i+4] - fter_x[i+0]) + (fter_x[i+3] - fter_x[i-1]) + (fter_x[i+2] - fter_x[i-2]) + (fter_x[i+1] - fter_x[i-3])
    angle = 180.*math.atan2(dy, dx)/math.pi

    if dy == 0.0 or dx == 0.0:
        input_angle = last_angle
    else:
        input_angle = 0.8*last_angle+0.2*angle
        last_angle = input_angle
    get_nav(img, -input_angle+180., i)
    
    if (i-start) % (total//100) == 0:
        print(str(round(100*(i-start)/total, 1))+'%')
"""