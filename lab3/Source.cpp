/**
CS585_Lab3.cpp
@author: Ajjen Joshi
@version: 1.0 9/17/2014

CS585 Image and Video Computing Fall 2014
Lab 3
--------------
This program introduces the following concepts:
a) Finding objects in a binary image
b) Filtering objects based on size
c) Obtaining information about the objects described by their contours
--------------
*/


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector> 
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

//Global variables
int thresh = 128;
int max_thresh = 255;

// Function header
void threshold_callback(int, void*);

// main function
int main()
{

	Mat src; Mat src_gray;

	// Load source image and convert it to gray
	src = imread("hand.jpg", 1);

	// Create Window and display source image
	namedWindow("Source", CV_WINDOW_AUTOSIZE);
	imshow("Source", src);

	// Convert image to gray
	// Documentation for cvtColor: http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html
	cvtColor(src, src_gray, CV_BGR2GRAY);
	// Blur the image
	// Documentation for blur: http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=blur#blur
	blur(src_gray, src_gray, Size(3, 3));

	// Create Trackbar
	// Documentation for createTrackbar: http://docs.opencv.org/modules/highgui/doc/user_interface.html?highlight=createtrackbar#createtrackbar
	// Example of adding a trackbar: http://docs.opencv.org/doc/tutorials/highgui/trackbar/trackbar.html
	createTrackbar("Threshold:", "Source", &thresh, max_thresh, threshold_callback, &src_gray);
	//threshold_callback(0, 0);

	// Wait until keypress
	waitKey(0);
	return(0);
}

// function threshold_callback (x is going to be a matrix that is passed in to x. to use it we will need to cast it)
void threshold_callback(int, void* x)
{
	Mat thres_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	cout << "Threshold: " << thresh << endl;
	// Convert into binary image using thresholding
	// Documentation for threshold: http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#threshold
	// Example of thresholding: http://docs.opencv.org/doc/tutorials/imgproc/threshold/threshold.html
	threshold(*(Mat *)x, thres_output, thresh, max_thresh, 0); // x is cast as a pointer to a matrix (Mat) and then we dereference the pointer

	// Create Window and display thresholded image
	namedWindow("Thres", CV_WINDOW_AUTOSIZE);
	imshow("Thres", thres_output);

	// Find contours
	// Documentation for finding contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
	findContours(thres_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat contour_output = Mat::zeros(thres_output.size(), CV_8UC3);
	cout << "The number of contours detected is: " << contours.size() << endl;

	// Find largest contour
	int maxsize = 0;
	int maxind = 0;
	Rect boundrec;

	if (contours.size() > 0){
		for (int i = 0; i < contours.size(); i++)
		{
			// Documentation on contourArea: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#
			double area = contourArea(contours[i]);
			if (area > maxsize) {
				maxsize = area;
				maxind = i;
				boundrec = boundingRect(contours[i]);
			}
		}

		// Draw contours
		// Documentation for drawing contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#drawcontours
		drawContours(contour_output, contours, maxind, Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
		drawContours(contour_output, contours, maxind, Scalar(0, 0, 255), 2, 8, hierarchy);
		// Documentation for drawing rectangle: http://docs.opencv.org/modules/core/doc/drawing_functions.html
		rectangle(contour_output, boundrec, Scalar(0, 255, 0), 1, 8, 0);

		cout << "The area of the largest contour detected is: " << contourArea(contours[maxind]) << endl;
		cout << "-----------------------------" << endl << endl;

		/// Show in a window
		namedWindow("Contours", CV_WINDOW_AUTOSIZE);
		imshow("Contours", contour_output);
	}
}
