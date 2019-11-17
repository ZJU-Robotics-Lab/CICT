#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
 
using namespace boost::filesystem;
namespace newfs = boost::filesystem;
using namespace cv;
 
int main(int argc, char ** argv)
{
    cv::Mat img_encode;
    img_encode = imread("../res/test.png", CV_LOAD_IMAGE_COLOR);
 
    //encode image and save to file
    std::vector<uchar> data_encode;
    imencode(".png", img_encode, data_encode);
    std::string str_encode(data_encode.begin(), data_encode.end());
 
    path p("../res/imgencode_cplus.txt");
    newfs::ofstream ofs(p);
    assert(ofs.is_open());
    ofs << str_encode;
    ofs.flush();
    ofs.close();
 
    //read image encode file and display
    newfs::fstream ifs(p);
    assert(ifs.is_open());
    std::stringstream sstr;
    while(ifs >> sstr.rdbuf());
    ifs.close();
 
    Mat img_decode;
    std::string str_tmp = sstr.str();
    std::vector<uchar> data(str_tmp.begin(), str_tmp.end());
    img_decode = imdecode(data, CV_LOAD_IMAGE_COLOR);
    imshow("pic",img_decode);
    cvWaitKey(10000);
 
    return 0;
}
