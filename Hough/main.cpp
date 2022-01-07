#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void hough_line_segments();

int main(){
	hough_line_segments();
	return 0;
}

void hough_line_segments(){
	Mat src = imread("card.bmp", IMREAD_GRAYSCALE);

	if(src.empty()){
		cerr << "File load failed" << endl;
		return;
	}

	Mat edge;
	Canny(src, edge, 200, 220);

	vector<Vec4i> lines;
	HoughLinesP(edge, lines, 1, CV_PI / 180, 90, 40, 2);

	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);

	for (Vec4i l : lines){
		line(dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, LINE_AA);
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}
