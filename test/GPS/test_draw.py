import time
import math
from PIL import Image
import matplotlib.pyplot as plt

def gps2xy(latitude, longtitude):
    
    	#remain to be done!!! figure out the formula meanings!!!
    	latitude = latitude * math.pi/180
    	longtitude = longtitude *math.pi/180
    
    	#the radius of the equator
    	radius = 6378137
    	#distance of the two poles
    	distance = 6356752.3142
    	#reference??
    	base = 30 * math.pi/180
    	
    	radius_square = pow(radius,2)
    	distance_square = pow(distance,2)
    	
    	e = math.sqrt(1 - distance_square/radius_square)
    	e2 = math.sqrt(radius_square/distance_square - 1)
    
    	cosb0 = math.cos(base)
    	N = (radius_square / distance) / math.sqrt( 1+ pow(e2,2)*pow(cosb0,2))
    	K = N*cosb0
    	
    	sinb = math.sin(latitude)
    	tanv = math.tan(math.pi/4 + latitude/2)
    	E2 = pow((1 - e*sinb) / (1+ e* sinb),e/2)
    	xx = tanv * E2;
    	
    	xc = K * math.log(xx)
    	yc = K * longtitude
    	return xc,yc



def avg(values):
    summ = sum(values)
    return summ/len(values)
  
def normal(values, is_y=False):
    avg_value = values[0]

    if is_y:
        avg_value = (11589596.333751025+11589605.59585501)/2
    else:
        avg_value = (3047362.377753752+3047362.981469983)/2
    #return [(item - avg_value)/(max_v - min_v) for item in values]
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
        self.MAX_V = 15.0
        
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


scale_x = 2.32
scale_y = 2.32
x_offset = 660
y_offset = 470
        
latitudes = []
longtitudes = []
with open('gps_log2.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        try:
            sp_line = line.split()
            latitude = float(sp_line[0])
            longtitude = float(sp_line[1])
            latitudes.append(latitude)
            longtitudes.append(longtitude)
        except:
            pass


xs = []
ys = []
for i in range(len(latitudes)):
    latitude = latitudes[i]
    longtitude = longtitudes[i]
    x,y = gps2xy(latitude, longtitude)
    xs.append(x)
    ys.append(y)

    
nx = normal(xs)
ny = normal(ys, True)

fter = Filter()
fter_x = []
fter_y = []
fter_v = []


for i in range(len(nx)):
    x, y, v = fter.step(nx[i], ny[i])

    _x = -x*scale_x  + x_offset
    _y = y*scale_y  + y_offset
    fter_x.append(_x)
    fter_y.append(_y)
    fter_v.append(v)

################################################
latitudes = []
longtitudes = []
with open('gps_log3.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        try:
            sp_line = line.split()
            latitude = float(sp_line[0])
            longtitude = float(sp_line[1])
            latitudes.append(latitude)
            longtitudes.append(longtitude)
        except:
            pass


xs = []
ys = []
for i in range(len(latitudes)):
    latitude = latitudes[i]
    longtitude = longtitudes[i]
    x,y = gps2xy(latitude, longtitude)
    xs.append(x)
    ys.append(y)

    
nx = normal(xs)
ny = normal(ys, True)

fter2 = Filter()
fter_x2 = []
fter_y2 = []
fter_v2 = []


for i in range(len(nx)):
    x, y, v = fter2.step(nx[i], ny[i])

    _x = -x*scale_x  + x_offset
    _y = y*scale_y  + y_offset
    fter_x2.append(_x)
    fter_y2.append(_y)
    fter_v2.append(v)


im = Image.open('map.png')
fig,ax = plt.subplots(1)
ax.imshow(im)
plt.axis('off')

plt.plot(fter_y,fter_x,'xkcd:black',linewidth=1)
plt.plot(fter_y2,fter_x2,'xkcd:black',linewidth=1)
plt.tight_layout()
fig.savefig('out.png', bbox_inches='tight', dpi=1000)
plt.show()