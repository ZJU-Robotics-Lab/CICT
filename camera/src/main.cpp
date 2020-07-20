#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mat_warper.h"
#include "mynteyed/camera.h"
#include "mynteyed/utils.h"
#include "util/cam_utils.h"

#define FRAMERATE 30
#define RESOLUTION StreamMode::STREAM_1280x720

namespace py = pybind11;
MYNTEYE_USE_NAMESPACE

class MyCamera{
public:
    MyCamera(){
        init();
    }
    void init(){
        cam = new Camera();
        dev_info = new DeviceInfo();
        OpenParams params(dev_info->index);
        params.framerate = FRAMERATE;
        params.color_mode = ColorMode::COLOR_RECTIFIED;
        params.dev_mode = DeviceMode::DEVICE_COLOR;
        params.stream_mode = RESOLUTION;
        cam->EnableImageInfo(true);
        cam->Open(params);
        cam->AutoWhiteBalanceControl(true);
        cam->AutoExposureControl(true);
        cam->SetExposureTime(1.0);
        cam->SetGlobalGain(1.0);
        bool is_left_ok = cam->IsStreamDataEnabled(ImageType::IMAGE_LEFT_COLOR);
        if (!cam->IsOpened() || !is_left_ok) {
            std::cerr << "Error: Open camera failed" << std::endl;
            return;
        }
        else{
            std::cout << "Open device success" << std::endl << std::endl;
        }
    }
    void close(){
        cam->DisableImageInfo();
        cam->Close();
        delete cam;
        delete dev_info;
    }
    py::array_t<unsigned char> getImage(){
        while(true){
            cam->WaitForStream();
            auto left_color = cam->GetStreamData(ImageType::IMAGE_LEFT_COLOR);
            if (left_color.img) {
                cv::Mat left = left_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
                cv::Mat dst;
                cv::cvtColor(left, dst, cv::COLOR_RGB2BGR);
                return cv_mat_uint8_3c_to_numpy(dst);
            }
            else{
                continue;
            }
        }
    }
private:
    Camera* cam;
    DeviceInfo *dev_info;
};

PYBIND11_MODULE(robot_camera, m) {
    m.doc() = "Python warper for MYNT EYE D-1000-120 camera";
    py::class_<MyCamera>(m, "Camera")
        .def(py::init())
        .def("close", &MyCamera::close)
        .def("getImage", &MyCamera::getImage);
}