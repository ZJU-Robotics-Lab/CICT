#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<fstream>
#include<iostream>

namespace py = pybind11;

/*
https://blog.csdn.net/u013701860/article/details/86313781
https://blog.csdn.net/u011021773/article/details/83188012
*/

py::array_t<float> calcMul(py::array_t<float>& input1, py::array_t<float>& input2) {

    // read inputs arrays buffer_info
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();

    if (buf1.size != buf2.size)
    {
        throw std::runtime_error("Input shapes must match");
    }

    // allocate the output buffer
    py::array_t<double> result = py::array_t<double>(buf1.size);



}

class Matrix
{
public:
    Matrix() {};
    Matrix(int rows, int cols) {
        this->m_rows = rows;
        this->m_cols = cols;
        m_data = new float[rows*cols];
    }
    ~Matrix() {};

private:
    int m_rows;
    int m_cols;
    float* m_data;

public:
    float* data() { return m_data; };
    int rows() { return m_rows; };
    int cols() { return m_cols; };

};




void save_2d_numpy_array(py::array_t<float, py::array::c_style> a, std::string file_name) {

    std::ofstream out;
    out.open(file_name, std::ios::out);
    std::cout << a.ndim() << std::endl;
    for (int i = 0; i < a.ndim(); i++)
    {
        std::cout << a.shape()[i] << std::endl;
    }
    for (int i = 0; i < a.shape()[0]; i++)
    {
        for (int j = 0; j < a.shape()[1]; j++)
        {
            if (j == a.shape()[1]-1)
            {
                //访问读取,索引 numpy.ndarray 中的元素
                out << a.at(i, j)<< std::endl;
            }
            else {
                out << a.at(i, j) << " ";
            }
        }
    }

}

PYBIND11_MODULE(robot_camera, m) {

    m.doc() = "Simple numpy demo";
    m.def("save_2d_numpy_array", &save_2d_numpy_array);
    //m.def("rgb_to_gray", &rgb_to_gray);
}
