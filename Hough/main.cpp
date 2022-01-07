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
	VideoCapture cap(0);

	if(!cap.isOpened()){
		cerr << "Camera open failed!" << endl;
		return ;
	}
	Mat src;
	while (true){
		cap >> src;
		if(src.empty())
			break;
		Mat sharp;
		Mat calc;
		Mat edge;
		GaussianBlur(src, sharp, Size(), 5);

		float alpha = 1.f;
		calc = (1+alpha) * src - alpha*sharp;

		Canny(calc, edge, 50, 200);

		vector<Vec4i> lines;
		HoughLinesP(edge, lines, 1, CV_PI / 180, 90, 40, 2);

		Mat dst;
		cvtColor(edge, dst, COLOR_GRAY2BGR);

		for (Vec4i l : lines){
			line(dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, LINE_AA);
		}

		imshow("src", src);
		imshow("calc", calc);
		imshow("dst", dst);

		if(waitKey(10)==27)
			break;
	}
	destroyAllWindows();
}
