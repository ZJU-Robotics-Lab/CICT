#define LINE_NUM 16
#define BEAM_NUM 2
#define BLOCK_NUM 12
#define POINT_NUM 384
#define RETRUN_DIM 4
#define BYTES_PER_BLOCK 100
#define AZIMUTH_RESOLUTION 0.01 //0.01 degree
#define DISTANCE_RESOLUTION 0.002 // 2mm

#include <iostream>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
namespace{
	const double LINE_ANGLE[LINE_NUM] = {-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15}; //degree
};
using namespace std;
namespace py = pybind11;

struct Channel{
	unsigned short distance;
	unsigned char reflectivity;
};

struct Beam{
	struct Channel channels[LINE_NUM];
};

struct Block{
	unsigned short flag;
	unsigned short azimuth;
	struct Beam beams[BEAM_NUM];
};

struct DataBag{
	struct Block blocks[BLOCK_NUM];
};

py::array_t<double> parseData(const char *data){
	struct DataBag dataBag;
	struct Block blocks[BLOCK_NUM];

	for(int blockId = 0; blockId < BLOCK_NUM; blockId++){
	    int offset = blockId * BYTES_PER_BLOCK;
	    unsigned short flag = (data[offset] & 0xFF) | (data[offset+1]<<8 & 0xFF00);
	    unsigned short azimuth = (data[offset+2] & 0xFF) | (data[offset+3]<<8 & 0xFF00);
	    blocks[blockId].flag = flag;
	    blocks[blockId].azimuth = azimuth;
	    offset += 4;

	    struct Beam beams[BEAM_NUM];
	    for(int beamId = 0; beamId < BEAM_NUM; beamId++){
			struct Channel channels[LINE_NUM];
			for (int channelId = 0; channelId < LINE_NUM; channelId++){
				unsigned short distance = (data[offset] & 0xFF) | (data[offset+1]<<8 & 0xFF00);
				unsigned char reflectivity = data[offset+2] & 0xFF;
				channels[channelId].distance = distance;
				channels[channelId].reflectivity = reflectivity;
				offset += 3;
				beams[beamId].channels[channelId] = channels[channelId];
			}// finish all channels
			blocks[blockId].beams[beamId] = beams[beamId];
	    }// finish all beams
		dataBag.blocks[blockId] = blocks[blockId];
	}// finish all blocks

	py::array_t<double> result = py::array_t<double>(RETRUN_DIM * POINT_NUM);
	py::buffer_info buff = result.request();
	double *ptr = (double *)buff.ptr;

	int pt_index = 0;
	for(int blockId = 0; blockId < BLOCK_NUM; blockId++){
		unsigned short _azimuth = dataBag.blocks[blockId].azimuth;
		// _azimuth = 0x63E0;//test
		double azimuth = static_cast<double>(_azimuth)*AZIMUTH_RESOLUTION * M_PI / 180.0;// degree
		// cout << "azimuth: 4.46246 --- " << azimuth << endl;//test
		for(int beamId = 0; beamId < BEAM_NUM; beamId++){
			for (int channelId = 0; channelId < LINE_NUM; channelId++){
				struct Channel channel = dataBag.blocks[blockId].beams[beamId].channels[channelId];
				unsigned short _distance = channel.distance;
				unsigned char _reflectivity = channel.reflectivity;
				// _distance = 0x07B6;//test
				// _reflectivity = 0x2A;//test
				double distance = static_cast<double>(_distance)*DISTANCE_RESOLUTION;// meter
				// cout << "distance: 3.948 --- " << distance <<" reflectivity 42 --- " << int(_reflectivity) << endl;//test
				double _z = distance * sin(LINE_ANGLE[channelId] * M_PI / 180.0);
				double xy = distance * cos(LINE_ANGLE[channelId] * M_PI / 180.0);
				double _x = xy * sin(azimuth);
				double _y = xy * cos(azimuth);
				ptr[pt_index*RETRUN_DIM + 0] = _x;
				ptr[pt_index*RETRUN_DIM + 1] = _y;
				ptr[pt_index*RETRUN_DIM + 2] = _z;
				ptr[pt_index*RETRUN_DIM + 3] = static_cast<double>(_reflectivity);
				pt_index += 1;
			}
		}
	}
	result.resize({POINT_NUM, RETRUN_DIM});
	return result;
}

PYBIND11_MODULE(velodyne, m) {
	m.doc() = "velodyne driver";
	m.def("parse_data", &parseData, "parse data function");
}