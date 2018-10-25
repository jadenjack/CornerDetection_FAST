#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

const char* WINDOW_NAME = "window";

const int PIXEL_COUNT = 16;
/*
Point point1(c + dx[0], r + dy[0]); ╩С
Point point5(c + dx[4], r + dy[4]); го
Point point9(c + dx[8], r + dy[8]); аб
Point point13(c + dx[12], r + dy[12]); ©Л
*/
const int dy[] = {-3,-3,-2,-1, 0, 1, 2, 3, 3, 3, 2, 1, 0,-1,-2,-3 };
const int dx[] = { 0, 1, 2, 3, 3, 3, 2, 1, 0,-1,-2,-3,-3,-3,-2,-1 };
const Scalar BLACK(0, 0, 0);
RNG rng(12345);

void detectFeatures(VideoCapture cap);
void showVideo(VideoCapture cap);
void writeVideo(const char* fileName, VideoCapture cap, int width, int height);

vector<Point> detectFeaturePoint(Mat frame, int threshold);
bool detect(Mat frame, int r, int c, int threshold);
bool inBoundary(Mat frame, int pixelY, int pixelX);
bool inBoundary(Mat frame, Point p);
void drawPoints(Mat frame, vector<Point> pointList);

int main(int argc, char**argv) {
	const char* target = "sample.mp4";
	VideoCapture capture(target);
	detectFeatures(capture);
	showVideo(capture);
	waitKey(0);
}

void detectFeatures(VideoCapture cap) {
	if (!cap.isOpened()) {
		cout << "capture is not opende" << endl;
		exit(1);
	}

	int threshold = 10;
	int n = 9;
	Mat frame;
	cap >> frame;
	
	namedWindow(WINDOW_NAME, 1);

	while (!frame.empty()) {
		vector<Point> pointList = detectFeaturePoint(frame, threshold);
		drawPoints(frame, pointList);
		imshow(WINDOW_NAME, frame);
		cap >> frame;
		waitKey(60);
	}
}

//Point(x,y) = Point(column, row) = at(y,x) = at(row,column)
//Point(c,r)
vector<Point> detectFeaturePoint(Mat frame, int threshold) {

	Mat grayFrame;
	vector<Point> pointList;
	bool detected;
	cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

	for (int r = 0; r < frame.rows; r++) {
		for (int c = 0; c < frame.cols; c++) {
			
			detected = detect(grayFrame,r,c,threshold);
			if (detected) {
				pointList.push_back(Point(c,r));
			}
		}
	}
	return pointList;
}

void drawPoints(Mat frame, vector<Point> pointList) {
	
	for (auto& p : pointList) {
		circle(frame, p, 3, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1);
	}
}

bool overThreshold(Mat frame, int pixelTarget, Point comparedPoint, int threshold)
{
	if (inBoundary(frame, comparedPoint)) {
		int pixel1 = frame.at<unsigned char>(comparedPoint);
		return abs(pixelTarget - pixel1) >= threshold;
	}
	return false;
}

//										row,  col
bool detect(Mat frame, int r, int c,int threshold) {
	
	int count=0;
	Point target(c, r);

	
	int pixelTarget = frame.at<unsigned char>(target);
	for (int i = 0; i < 4; i++) {
		Point compare(c + dx[4 * i], r + dy[4 * i]);
		if (overThreshold(frame, pixelTarget, compare, threshold))
			count++;
	}

	return count>=3;
}

bool inBoundary(Mat frame, Point p) {
	return inBoundary(frame, p.y, p.x);
}

bool inBoundary(Mat frame, int pixelY, int pixelX) {
	return pixelY >= 0 && pixelX >= 0 && pixelY < frame.rows && pixelX < frame.cols;
}

void showVideo(VideoCapture cap) {
	if (!cap.isOpened()) {
		cout << "capture is not opende" << endl;
		exit(1);
	}

	Mat frame, grayFrame;
	cap >> frame;

	namedWindow(WINDOW_NAME, 1);
	while (!frame.empty()) {
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		imshow(WINDOW_NAME, grayFrame);
		cap >> frame;
		waitKey(60);
	}
}

void writeVideo(const char* fileName, VideoCapture cap,int width, int height) {
	VideoWriter video(fileName, CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(width, height));
	Mat frame;
	cap >> frame;
	while (!frame.empty()) {
		imshow(WINDOW_NAME, frame);
		cap >> frame;
		waitKey(60);
	}
	cap.release();
	video.release();
	destroyAllWindows();
}

void openCVFAST() {
	const char* target = "sample.mp4";
	VideoCapture capture(target);
	Mat src;
	capture >> src;
	cout << " Image size :" << src.rows << " " << src.cols << "\n";
	vector<KeyPoint> keypointsD;
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
	vector<Mat> descriptor;

	detector->detect(src, keypointsD, Mat());
	drawKeypoints(src, keypointsD, src);
	imshow("keypoints", src);
	waitKey();
}