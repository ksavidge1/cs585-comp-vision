/**
hw3FishFinal.cpp
@author: Kyle Savidge, Mingxiang Cai, Huai Chun Shih
@version: 1.0 10/05/2016

CS585 Image and Video Computing Fall 2014
HW 3
--------------
*/


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include<cmath>
#include <chrono>
#include <thread>



using namespace cv;
using namespace std;

//Global variables
int thresh = 1;
int max_thresh = 100;

// Function header
void segmentaion(Mat &src, Mat &dst);
void adThres(Mat& src, Mat& dst, int bs, int c);
double adThelper(Mat& src, int x, int y, int bs, int c);
void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs,	vector < vector<Point2i > > &boundaries);
void FindBoundary(const Mat &label_image, vector < vector<Point2i> > &boundaries, int &label_count, Rect &rect);
void drawBoundaries(Mat &output, vector < vector<Point2i> > &boundaries);
void computeCircularities(vector < vector<Point2i> > &blobs, vector < double > &circularities);
void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float);
double getOrientation(const vector<Point> &pts, Mat &img);
void FilterFish(Mat &src, Mat &dst);
void rgbToBlueGrayScal(Mat& src, Mat& dst);
void rgbToRedGrayScal(Mat& src, Mat& dst);


// main function
int main()
{

	Mat src;
	Mat src_gray;

	//adThres(src_gray, thres_output, 11, -7);
	//adaptiveThreshold(src_gray, thres_output, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 121, -7);

	namedWindow("Region1", WINDOW_NORMAL);
	namedWindow("Region2", WINDOW_NORMAL);
	namedWindow("original", WINDOW_AUTOSIZE);

	while(true){
	for(int i = 3; i <= 301; i++)
	{
		cout << i << endl;
		char name[100];
		sprintf(name, "CS585-AquariumImages/aquarium_frame%09d.ppm.png", i);
		src = imread(name, 1);

		Mat src1 = src(Rect(src.cols-450,src.rows-750,400,700));
		Mat dst1 = src1.clone();
		rgbToBlueGrayScal(src1, dst1);
		FilterFish(dst1, dst1);
		segmentaion(dst1, dst1);

		Mat src2 = src(Rect(400, 350, 400, 100));
		Mat dst2 = src2.clone();
		rgbToRedGrayScal(src2, dst2);
		//cvtColor(dst2, dst2, CV_RGB2GRAY);
		FilterFish(dst2, dst2);
		segmentaion(dst2, dst2);

		resize(src, src, Size(), 0.5, 0.5);
		//resize(dst, dst, Size(), 0.5, 0.5);
		imshow("Region1", dst1);
		imshow("Region2", dst2);
		imshow("original", src);
		//slow down motion
		//std::this_thread::sleep_for(std::chrono::milliseconds(200));
		if(waitKey(1)==27)
			exit(0);
	}
	}

	// Wait until keypress

	waitKey(0);
	return(0);
}

//get a grey scale image with fish filtered
void FilterFish(Mat &src, Mat &dst)
{
	Mat thres_output;
	Mat img = src;

	// Convert into binary image using thresholding
	// Documentation for threshold: http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#threshold
	// Example of thresholding: http://docs.opencv.org/doc/tutorials/imgproc/threshold/threshold.html

	//use 121 -2 for cell
	//threshold(*(Mat *)x, thres_output, thresh, max_thresh, 1); // x is cast as a pointer to a matrix (Mat) and then we dereference the pointer
	//adThres(img, thres_output, thresh+(thresh%2)+1, -7);
	blur(img, img, Size(2, 2));
	adaptiveThreshold(img, thres_output, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 13, -6);

	//initialize element
	int ed_type;
	int ed_elem = 2;

	int erosion_size = 1;
	int dilate_size = 1;

	if( ed_elem == 0 ){ ed_type = MORPH_RECT; }
	else if( ed_elem == 1 ){ ed_type = MORPH_CROSS; }
	else if( ed_elem == 2) { ed_type = MORPH_ELLIPSE; }

	Mat erosion_elem = getStructuringElement( ed_type,
																			 Size( 2*erosion_size + 1, 2*erosion_size+1 ),
																			 Point( erosion_size, erosion_size ) );
	Mat dilate_elem = getStructuringElement( ed_type,
																			 Size( 2*dilate_size + 1, 2*dilate_size+1 ),
																			 Point( dilate_size, dilate_size ) );

	/// Apply the erosion operation
	//d= 3~4
	for (int i = 0; i<2; i++) dilate( thres_output, thres_output, dilate_elem );
	for (int i = 0; i<2; i++) erode(thres_output, thres_output, erosion_elem);

	dst = thres_output;
}

// function threshold_callback (x is going to be a matrix that is passed in to x. to use it we will need to cast it)
void segmentaion(Mat &src, Mat &dst)
{
	Mat thres_output = src;
	Mat img = src;

  //apply the filter funtion
	//FilterFish(img , thres_output);

	// eliminate boundary white
	for (int i = 0; i < thres_output.rows; i ++) {
			thres_output.at<uchar>(i, thres_output.cols-1) = 0;
			thres_output.at<uchar>(i, 0) = 0;
	}

	for (int i = 0; i < thres_output.cols; i ++) {
			thres_output.at<uchar>(thres_output.rows-1, i) = 0;
			thres_output.at<uchar>(0, i) = 0;
	}


		Mat output = Mat::zeros(img.size(), CV_8UC3);
		Mat binary;
    vector < vector<Point2i > > blobs;
		vector < vector<Point2i > > boundaries;
		vector < double > circularities;
		vector < double > orientations;

    threshold(thres_output, binary, 0.0, 1.0, cv::THRESH_BINARY);

    FindBlobs(binary, blobs, boundaries);

		computeCircularities(blobs, circularities);

    // Randomy color the blobs
    for(size_t i=0; i < blobs.size(); i++)

		{
        unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));

        for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;

            output.at<Vec3b>(y,x)[0] = b;
            output.at<Vec3b>(y,x)[1] = g;
            output.at<Vec3b>(y,x)[2] = r;

        }

				//get orientation
				orientations.push_back(getOrientation(boundaries[i], output));

				//output data for each object
				cout << "region " << i << " : ";
				cout << "area: " << blobs[i].size() << "   ";
				cout << "circularity: " << circularities[i] << "   ";
				cout << "compactness: " << blobs[i].size()/boundaries[i].size() << "   ";
				cout << "orientation: " << orientations[i] << "   ";
				cout << "# of boundary pixels: "  << boundaries[i].size() << endl;

    }

		drawBoundaries(output, boundaries);
		dst = output;
}


//
void FindBlobs(const Mat &binary, vector < vector<Point2i> > &blobs, 	vector < vector<Point2i > > &boundaries)
{
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

            Rect rect;
            floodFill(label_image, Point(x,y), label_count, &rect, 0, 0, 4);

            vector <Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back(Point2i(j,i));
                }
            }

						if (blob.size()>10)
						{
        				blobs.push_back(blob);
								cout << label_count << endl;
								FindBoundary(label_image, boundaries, label_count, rect);
						}


            label_count++;
        }
    }
}

//function that stores the boundaries pixels for each object into boundary vector
void FindBoundary(const Mat &label_image, vector < vector<Point2i> > &boundaries, int &label_count, Rect &rect)
{
	//obtain  a starting  point;
	int sx;
	int sy;
	for(int i=rect.y; i < (rect.y+rect.height); i++) {
			int *row2 = (int*)label_image.ptr(i);
			for(int j=rect.x; j < (rect.x+rect.width); j++) {
					if(row2[j] != label_count) {
							continue;
					}

					sx = j;
					sy = i;
					i =  rect.y+rect.height;
					break;
			}
	}

	//cout << "this is the starting point: " << sx << " " << sy <<  endl;

	vector <Point2i> boundary;
	boundary.push_back(Point2i(sx,sy));

	int cx = sx;
	int cy = sy;
	int nx = sx-1;
	int ny = sy;
	int lx;
	int ly;

	// //show image aorund the start point
	// for (int j = sy-3; j<=sy+3; j++) {
	// 	int *row = (int*)label_image.ptr(j);
	// 	 for (int i = sx -3; i<= sx+3; i++){
	// 		 cout << row[i] << " ";
	// 	 }
	// 	 cout << endl;
	//  }

	int *row = (int*)label_image.ptr(ny);
	int label = row[nx];


	do
	{
			//move clockwise and recoed the last position until hitting next object pixel;
			while (label != label_count)
			{
				  //W
					if ((nx == cx-1) && (ny == cy))
					{
							lx = nx;
							ly = ny;
							ny--;
							int *row = (int*)label_image.ptr(ny);
							label = row[nx];
							//cout << "W " << row[nx]<< endl;
							continue;
					}
					//NW
					if ((nx == cx-1) && (ny == cy-1))
					{
							lx = nx;
							ly = ny;
							nx++;
							int *row = (int*)label_image.ptr(ny);
							//cout << "NW " << row[nx]<< endl;
							label = row[nx];
							continue;
					}
					//N
					if ((nx == cx) && (ny == cy-1))
					{
							lx = nx;
							ly = ny;
							nx++;
							int *row = (int*)label_image.ptr(ny);
							//cout << "N " << row[nx]<< endl;
							label = row[nx];
							continue;
					}
					//NE
					if ((nx == cx+1) && (ny == cy-1))
					{
							lx = nx;
							ly = ny;
							ny++;
							int *row = (int*)label_image.ptr(ny);
							label = row[nx];
							continue;
					}
					//E
					if ((nx == cx+1) && (ny == cy))
					{
							lx = nx;
							ly = ny;
							ny++;
							int *row = (int*)label_image.ptr(ny);
							label = row[nx];
							continue;
					}
					//SE
					if ((nx == cx+1) && (ny == cy+1))
					{
							lx = nx;
							ly = ny;
							nx--;
							int *row = (int*)label_image.ptr(ny);
							label = row[nx];
							continue;
					}
					//S
					if ((nx == cx) && (ny == cy+1))
					{
							lx = nx;
							ly = ny;
							nx--;
							int *row = (int*)label_image.ptr(ny);
							label = row[nx];
							continue;
					}
					//SW
					if ((nx == cx-1) && (ny == cy+1))
					{
							lx = nx;
							ly = ny;
							ny--;
							int *row = (int*)label_image.ptr(ny);
							label = row[nx];
							continue;
					}


			}

			//debug use
			// cout << "found: " << nx << " " << ny << " ";
			// cout << "last point: " << lx << " " << ly << endl;

			//next boundary pixel found!
			boundary.push_back(Point2i(nx,ny));
			cx = nx;
			cy = ny;
			nx = lx;
			ny = ly;
			int *row = (int*)label_image.ptr(ny);
			label = row[nx];


	} while (!(cx == sx && cy == sy));

	//save boundary for a object
	boundaries.push_back(boundary);
}

//draw boundary
void drawBoundaries(Mat &output, vector < vector<Point2i> > &boundaries)
{
	for(size_t i=0; i < boundaries.size(); i++)
	{

					for(size_t j=0; j < boundaries[i].size(); j++) {
							int x = boundaries[i][j].x;
							int y = boundaries[i][j].y;

							output.at<Vec3b>(y,x)[0] = 0;
							output.at<Vec3b>(y,x)[1] = 0;
							output.at<Vec3b>(y,x)[2] = 255;

					}
		}
}

//compute circularity of each subject and save it in circularities
void computeCircularities(vector < vector<Point2i> > &blobs, vector < double > &circularities)
{
	for(size_t i=0; i < blobs.size(); i++)
	{
			int xsum = 0;
			int ysum = 0;
			double xmean;
			double ymean;
			double a = 0.0;
			double b = 0.0;
			double c = 0.0;

			for(size_t j=0; j < blobs[i].size(); j++) {
					xsum += blobs[i][j].y;
					ysum += blobs[i][j].x;
			}

			xmean = xsum /  blobs[i].size();
			ymean = ysum /  blobs[i].size();

			for(size_t j=0; j < blobs[i].size(); j++) {
					a += pow(blobs[i][j].y-xmean, 2);
					c += pow(blobs[i][j].x-ymean, 2);
					b += (blobs[i][j].y-xmean)*(blobs[i][j].x-ymean);
			}

			double r = sqrt(pow(b, 2)+ pow(a-c, 2));
			double ratio = ((a+c)*r-pow(a-c, 2)-pow(b, 2))/((a+c)*r+pow(a-c, 2)+pow(b, 2));

			//std::cout << atan(b/(a-c)) / 2 * 180 / M_PI << std::endl;
			circularities.push_back(ratio);
	}
}

//orientation functions from pca method http://docs.opencv.org/3.1.0/d1/dee/tutorial_introduction_to_pca.html need processed by skin detect first
void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
    double angle;
    double hypotenuse;
    angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
    //    double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
    //    cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, colour, 1, CV_AA);
    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, CV_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, CV_AA);
}

double getOrientation(const vector<Point> &pts, Mat &img)
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
    //Store the center of the object
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                       static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; ++i)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
    }
    // Draw the principal components
    //circle(img, cntr, 3, Scalar(255, 0, 255), 2);
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    //drawAxis(img, cntr, p1, Scalar(0, 255, 0), 1);
    //drawAxis(img, cntr, p2, Scalar(255, 255, 0), 5);
    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
    return angle;
}

//rgb to blue
void rgbToBlueGrayScal(Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            //get the vetor of RGB
            Vec3b intensity = src.at<Vec3b>(i, j);
            int B = intensity[0];
            int G = intensity[1];
            int R = intensity[2];
            dst.at<uchar>(i,j) = B;//myMax(R,G,B);
        }
    }
}

void rgbToRedGrayScal(Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            //get the vetor of RGB
            Vec3b intensity = src.at<Vec3b>(i, j);
            int B = intensity[0];
            int G = intensity[1];
            int R = intensity[2];
            dst.at<uchar>(i,j) = R;//myMax(R,G,B);


        }
    }
}

//function that does mean adaptive Threshold with block size = bs, constant = cons;
void adThres(Mat& src, Mat& dst, int bs, int c)
{
	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++){
			  if (dst.at<uchar>(i,j) > adThelper(src, i, j, bs, c))
					dst.at<uchar>(i, j) = 255;
				else
					dst.at<uchar>(i, j) = 0;
			}
}

double adThelper(Mat& src, int x, int y, int bs, int c)
{
	  int a = max(x-(bs-1)/2, 0);
		int b = max(y-(bs-1)/2, 0);
		double T = 0;
		for (int i = a; i < min(a+bs, src.rows); i++)
			for (int j =b; j < min(b+bs, src.cols); j++)
					T += src.at<uchar>(i,j);
		T = T/(bs*bs)-c;
		return T;
}
