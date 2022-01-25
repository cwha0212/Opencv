#ifndef __LINE_DETECTOR_H__
#define __LINE_DETECTOR_H__

#include <iostream>
#include <cstring>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class LineDetector {
	private:
		double img_size, img_center;
		double left_m, right_m;
		Point left_b, right_b;
		bool left_detect = false, right_detect = false;

		double bottom_width = 0.85;
		double top_width = 0.07;
		double trap_height = 0.4;

	public:
		Mat filter_colors(Mat img);
		Mat limit_region(Mat img);
		vector<Vec4i> houghline(Mat img);
		vector<vector<Vec4i>> separateline(Mat img, vector<Vec4i> lines);
		vector<Point> regression(vector<vector<Vec4i>> separated_line, Mat img);
		string predictDir();
		Mat drawline(Mat img, vector<Point> lane, string dir);
};

#endif
