#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

//function headers
/**
Function that visualizes the optical flow, as lines drawn on subsampled pixels
@param flow 2	channel matrix of the same size of the input image, contains the flow vector for each pixel
@param cflowmap image on which you will draw the sub-sampled flow vectors
@param step		stepsize to subsample the original image
@param color	color with which the flow vectors are drawn
*/
//void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color);

//global variables
Mat frame; // current frame

int main(int argc, char** argv)
{
	VideoCapture cap("1.MP4"); // open the mp4
	//cap.set(CV_CAP_PROP_POS_FRAMES, 18000); //jump to 10 min, given fps = 30
	cap.set(CV_CAP_PROP_POS_FRAMES, 21800); //jump to landing of copter 1
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	namedWindow("Original video", CV_WINDOW_AUTOSIZE);

	//Initialize optical flow variables 
	Mat flow, cflow;
	Mat gray, prevgray, uflow;

	while (cap.isOpened())
	{
		Mat frame;
		//double msec = cap.get(CV_CAP_PROP_POS_MSEC); //method for getting the current video time in msec
		if (!cap.read(frame))  //break loop if can't read the frame
		{
			cout << "Cannot read the video file";
			break;
		}

		//Play the video
		imshow("Original video", frame);

		/*//Optical Flow
		Mat gray;
		cvtColor(frame, gray, CV_BGR2GRAY);
		if (!prevgray.empty())
		{
			//call Farneback's optical flow algorithm
			//Documentation: http://docs.opencv.org/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
			//Link to Farneback's paper: http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdf
			//calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 1, 45, 3, 5, 1.2, 0);
			calcOpticalFlowFarneback(prevgray, gray, uflow, .1, 1, 10, 1, 5, 1.1, 0);

			//cflow is the Mat where you will draw your optical flow vectors
			cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
			drawOptFlowMap(uflow, cflow, 64, Scalar(0, 255, 0));
			imshow("Optical Flow", cflow);
		}*/

		//Convert to grayscale, apply adaptive threshold
		Mat thresh;
		cvtColor(frame, gray, CV_BGR2GRAY);
		adaptiveThreshold(gray, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 5, 2);
		imshow("Thresholded", thresh);

		if (waitKey(30) == 27)  //wait for 'esc' key to be pressed
		{
			break;
		}
		swap(prevgray, gray);
	}
	return 0;
}

/*void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color)
{
	for (int y = 0; y < cflowmap.rows; y += step)
	for (int x = 0; x < cflowmap.cols; x += step)
	{
		const Point2f& fxy = flow.at<Point2f>(y, x);
		line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
			color);
		circle(cflowmap, Point(x, y), 2, color, -1);
	}
}*/


