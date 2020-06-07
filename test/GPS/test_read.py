import math

read_lines = [
    'Net SignalQuality 17\r\n',
    'Net Network Operator = Unknown\r\n',
    '$GPGGA,075719.00,3015.77096106,N,12006.99580974,E,1,17,1.2,38.6041,M,7.7018,M,,*68\r\n',
    'Net SignalQuality 17\r\x1aNet Network Operator = Unknown\r\n',
    'Net SignalQuality 17\r\n',
    'Net Network Operator = Unknown\r\n',
    '$GPGGA,075720.00,3015.77094785,N,12006.99571778,E,1,18,1.1,38.0302,M,7.7018,M,,*6F\r\n',
    'Net SignalQuality 17\r\n',
    'Net Network Operator = Unknown\r\n',
    'Net SignalQuality 17\r\n',
    '$GPGGA,075721.00,3015.77093772,N,12006.99562582,E,1,18,1.1,37.4653,M,7.7018,M,,*6E\r\n',
    'Net Network Operator = Unknown\r\n',
    'Net SignalQualZW\x8117\r\n',
    'Net Network Operator = Unknown\r\n',
    '$GPGGA,075722.00,3015.77093707,N,12006.99554921,E,1,18,1.1,36.9856,M,7.7018,M,,*68\r\n',
    'Net SignalQuality 17\r\n',
    'Net Network Operator = Unknown\r\n',
    'Net SignalQuality 17\r\n',
    'Net Network Operator = Unknown\r\n',
    '$GPGGA,075723.00,3015.77093691,N,12006.99547833,E,1,18,1.1,36.5309,M,7.7018,M,,*6A\r\n',
    'Net SignalQuality 17\r\n',
    'Net Network Operator = Unknown\r\n',
    '$GPGGA,075724.00,3015.77093227,N,12006.99539879,E<1,18,1.1,36.0461,M,7.7018,M,,*6F\r\n',
    'Net SignalQuality 17\r\n',
    'Net Network Operator = Unknown\r\n'
]

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

def parseGPS(line):
    data = line.split(',')
    #print(data)
    latitude = data[2]	
    longtitude = data[4]
    #num_star = data[7]
    #hdop = data[8]
    #convert degree+minute to degree
	#lantitude: DDmm.mm
	#longtitude: DDDmm.mm
    lan_degree = latitude[:2]
    lan_minute = latitude[2:]
    latitude = float(lan_degree) + float(lan_minute)/60

    long_degree = longtitude[:3]
    long_minute = longtitude[3:]
    longtitude = float(long_degree) + float(long_minute)/60
    return latitude,longtitude
    
for line in read_lines:
    if line[:6] == '$GPGGA':
        latitude,longtitude = parseGPS(line)
        print(latitude,longtitude)
        xc,yc = gps2xy(latitude, longtitude)