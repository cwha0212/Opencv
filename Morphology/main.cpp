#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void morphology(){
	VideoCapture cap(0);

	while (true){
		Mat frame;
		Mat src;
		cap >> frame;
		cvtColor(frame, src, COLOR_BGR2GRAY);

		Mat bin;
		threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

		Mat dst1, dst2, dst3;
		morphologyEx(bin, dst1, MORPH_OPEN, Mat());
		morphologyEx(bin, dst2, MORPH_CLOSE, Mat());
		morphologyEx(bin, dst3, MORPH_GRADIENT, Mat());

		imshow("frame",src);
		imshow("bin", bin);
		imshow("OPEN", dst1);
		imshow("CLOSE", dst2);
		imshow("GRADIENT", dst3);

		if(waitKey(10) ==27)
			break;
	}
	destroyAllWindows();
}

int main(){
	morphology();

	return 0;
}
