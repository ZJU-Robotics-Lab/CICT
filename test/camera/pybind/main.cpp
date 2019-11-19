#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mat_warper.h"

namespace py = pybind11;

py::array_t<unsigned char> test_rgb_to_gray(py::array_t<unsigned char>& input) {
    cv::Mat img_rgb = numpy_uint8_3c_to_cv_mat(input);
    cv::Mat dst;
    cv::cvtColor(img_rgb, dst, cv::COLOR_RGB2GRAY);
    return cv_mat_uint8_1c_to_numpy(dst);
}

py::array_t<unsigned char> test_read_img() {
    cv::Mat img_rgb = cv::imread("1528.jpg", cv::IMREAD_COLOR);
    cv::Mat dst;
    cv::cvtColor(img_rgb, dst, cv::COLOR_RGB2BGR);
    return cv_mat_uint8_3c_to_numpy(dst);

}

PYBIND11_MODULE(robot_camera, m) {
    m.doc() = "Simple opencv demo";
    m.def("test_rgb_to_gray", &test_rgb_to_gray);
    m.def("test_read_img", &test_read_img);
}