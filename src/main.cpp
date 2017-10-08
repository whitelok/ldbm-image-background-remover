#include <iostream>
#include <vector>
#include <string.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "LDBM.h"

using namespace std;
using namespace cv;

void showImg(std::string name, cv::Mat mat)
{
    cv::namedWindow(name);
    cv::imshow(name, mat);
}

void saveImg(std::string name, cv::Mat mat)
{
    std::string filePath;
    filePath = std::string("tempPicture_") + name + ".png";
    cv::imwrite(filePath, mat);
}

void printUsage()
{
    printf("Usage: LDBMImageBackgroundRemover /path/of/image /path/of/image_tag lamda \n");
}

string type2str(int type) {
    string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    r += "C";
    r += (chans+'0');
    return r;
}

Mat cutoutImg(Mat input, cv::Mat mask, double lamda)
{
    Mat input_bgra;
    cv::cvtColor(input, input_bgra, CV_BGR2BGRA);
    for (int y = 0; y < mask.rows; ++y){
        for (int x = 0; x < mask.cols; ++x){
            cv::Vec4b & pixel = input_bgra.at<cv::Vec4b>(y, x);
            if(mask.at<double>(y,x) < lamda){
                pixel[3] = 0;
            }else{
                continue;
            }
        }
    }
    return input_bgra;
}

int main(int argc, char *argv[])
{
    if(argc < 3){
        printUsage();
        exit(1);
    }else{
        Mat mat, trimap;
        mat = imread(argv[1], IMREAD_COLOR);
        trimap = imread(argv[2], IMREAD_GRAYSCALE);
        double lamda = 0.5;
        if (argc == 4){
            lamda = atof(argv[3]);
        }
        
        Mat alpha =  LBDM_Matting(mat, trimap);
        Mat cutout = cutoutImg(mat, alpha, lamda);
        
        saveImg("cutout", cutout);
    }
    exit(0);
}
