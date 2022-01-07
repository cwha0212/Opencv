#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void canny_edge(){
	Mat src = imread("card.bmp", IMREAD_GRAYSCALE);

	if (src.empty()){
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst1, dst2;
	Canny(src, dst1, 50, 100);
	Canny(src, dst2, 210, 220);

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);

	waitKey();
	destroyAllWindows();
}

int main(){
	canny_edge();
	return 0;
}
