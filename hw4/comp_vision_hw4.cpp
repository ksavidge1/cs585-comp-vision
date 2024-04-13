/**
Kyle Savidge
HW4
Attempts to segment datasets, and perform component analysis on the segmented images
*/

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

//Global variables
int thresh = 55;
int max_thresh = 255;

/**
Function that detects blobs from a binary image
@param binaryImg The source binary image (binary image contains pixels labeled 0 or 1 (not 255))
*/
void FindBinaryLargeObjects(const Mat& binaryImg);

//threshold callback function header
void createBlue(Mat& input, Mat& dst);


int main()
{
	// read image as grayscale
	Mat img = imread("IMG_5216.jpg", 0);
	if (!img.data) {
		cout << "File not found" << std::endl;
		return -1;
	}

	img.convertTo(img, CV_8UC3);
	//pre-process with blur
	blur(img, img, Size(3, 3));

	// create windows
	namedWindow("Original");
	namedWindow("bluechannel");

	int x = img.cols;
	int y = img.rows;
	Mat blueimg(Size(x, y), CV_8UC1);

	createBlue(img, blueimg);

	imshow("bluechannel", blueimg);

	waitKey(0);
	return 0;
}

void createBlue(Mat& input, Mat& dst){
	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			Vec3b intensity = input.at<Vec3b>(i, j);
			uchar B = intensity[0];
			dst.at<uchar>(i, j) = B;
		}
	}
}
	