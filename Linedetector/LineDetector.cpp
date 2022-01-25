#include <iostream>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "LineDetector.h"

Mat LineDetector::filter_colors(Mat img){
	// 흰색과 노랑색 차선 필터링//
	Mat output;
	UMat img_hsv, white_mask, white_img, yellow_mask, yellow_img;
	img.copyTo(output);

	Scalar lower_white = Scalar(200,200,200);
	Scalar upper_white = Scalar(255,255,255);
	Scalar lower_yellow = Scalar(10,100,100);
	Scalar upper_yellow = Scalar(40,255,255);

	inRange (output, lower_white, upper_white, white_mask);
	bitwise_and (output, output, white_img, white_mask);

	// 흰색 처리 끝 //

	cvtColor(output, img_hsv, COLOR_BGR2HSV);

	inRange(img_hsv, lower_yellow, upper_yellow, yellow_mask);
	bitwise_and (output, output, yellow_img, yellow_mask);

	// 노란색 처리 끝 //

	addWeighted(white_img, 1.0, yellow_img, 1.0, 0.0, output);
	return output;
}

Mat LineDetector::limit_region(Mat img){
	// 관심영역 선정 //
	int width = img.cols;
	int height = img.rows;

	Mat output;
	Mat mask = Mat::zeros(height,width, CV_8UC1);

	Point points[4]{
		Point((width * (1-bottom_width))/2, height), Point((width * (1-top_width))/2, height-height*trap_height), Point(width - (width * (1-top_width))/2, height-height*trap_height), Point(width - (width * (1-bottom_width))/2, height)
	};

	fillConvexPoly(mask, points, 4, Scalar(255,0,0));

	bitwise_and(img, mask, output);
	return output;
}

vector<Vec4i> LineDetector::houghline(Mat img){
	// 관심영영의 선 추출 //
	vector<Vec4i> line;

	HoughLinesP(img, line, 1, CV_PI/180, 20, 10, 20);
	return line;
}

vector<vector<Vec4i>> LineDetector::separateline(Mat img, vector<Vec4i> lines) {
	// 왼쪽, 오른쪽 선 추출 //	
	vector<vector<Vec4i>> output(2);
	Point p1, p2;
	vector<double> slopes;
	vector<Vec4i> final_lines, left_lines, right_lines;
	double slope_thresh = 0.3;

	// 검출된 직선들의 기울기를 계산 //
	for (int i = 0; i < lines.size(); i++) {
		Vec4i line = lines[i];
		p1 = Point(line[0], line[1]);
		p2 = Point(line[2], line[3]);

		double slope;
		if (p2.x - p1.x == 0)
			slope = 999.0;
		else
			slope = (p2.y - p1.y) / (double)(p2.x - p1.x);

		if (abs(slope) > slope_thresh) {
			slopes.push_back(slope);
			final_lines.push_back(line);
		}
	}

	img_center = double (img.cols / 2);

	for (int i = 0; i < final_lines.size(); i++) {
		p1 = Point(final_lines[i][0], final_lines[i][1]);
		p2 = Point(final_lines[i][2], final_lines[i][3]);

		if (slopes[i] > 0 && p1.x > img_center && p2.x > img_center) {
			right_detect = true;
			right_lines.push_back(final_lines[i]);
		}
		else if (slopes[i] < 0 && p1.x < img_center && p2.x < img_center ) {
			left_detect = true;
			left_lines.push_back(final_lines[i]);
		}
	}

	output[0] = right_lines;
	output[1] = left_lines;
	return output;
}

vector<Point> LineDetector::regression(vector<vector<Vec4i>> separated_line, Mat img) {
	// 가장 적당한 선 추출 //
	vector<Point> output(4);
	Point p1, p2, p3, p4;
	Vec4d left_line, right_line;
	vector<Point> left_points, right_points;

	if (right_detect) {
		for (auto i : separated_line[0]) {
			p1 = Point(i[0], i[1]);
			p2 = Point(i[2], i[3]);

			right_points.push_back(p1);
			right_points.push_back(p2);
		}

		if (right_points.size() > 0) {
			fitLine(right_points, right_line, DIST_L2, 0, 0.01, 0.01);
			right_m = right_line[1] / right_line[0];  //기울기
			right_b = Point(right_line[2], right_line[3]);
		}
	}

	if (left_detect) {
		for (auto j : separated_line[1]) {
			p3 = Point(j[0], j[1]);
			p4 = Point(j[2], j[3]);

			left_points.push_back(p3);
			left_points.push_back(p4);
		}

		if (left_points.size() > 0) {
			fitLine(left_points, left_line, DIST_L2, 0, 0.01, 0.01);

			left_m = left_line[1] / left_line[0];  //기울기
			left_b = Point(left_line[2], left_line[3]);
		}
	}

	//y = m*x + b  --> x = (y-b) / m
	int y1 = img.rows;
	int y2 = 470;

	double right_x1 = ((y1 - right_b.y) / right_m) + right_b.x;
	double right_x2 = ((y2 - right_b.y) / right_m) + right_b.x;

	double left_x1 = ((y1 - left_b.y) / left_m) + left_b.x;
	double left_x2 = ((y2 - left_b.y) / left_m) + left_b.x;

	output[0] = Point(right_x1, y1);
	output[1] = Point(right_x2, y2);
	output[2] = Point(left_x1, y1);
	output[3] = Point(left_x2, y2);

	return output;
}

string LineDetector::predictDir() {
	// 방향 예측 //

	string output;
	double x, threshold = 10;

	//두 차선이 교차하는 지점 계산
	x = (double)(((right_m * right_b.x) - (left_m * left_b.x) - right_b.y + left_b.y) / (right_m - left_m));

	if (x >= (img_center - threshold) && x <= (img_center + threshold))
		output = "Straight";
	else if (x > img_center + threshold)
		output = "Right Turn";
	else if (x < img_center - threshold)
		output = "Left Turn";

	return output;
}

Mat LineDetector::drawline(Mat img, vector<Point> lane, string dir) {
	// 색상으로 길 방향 보이기 //
	vector<Point> poly_points;
	Mat output;
	img.copyTo(output);

	poly_points.push_back(lane[2]);
	poly_points.push_back(lane[0]);
	poly_points.push_back(lane[1]);
	poly_points.push_back(lane[3]);

	fillConvexPoly(output, poly_points, Scalar(230,0,30), LINE_AA, 0);
	addWeighted(output, 0.3, img, 0.7, 0, img);
	
	Size sizeImg = img.size();
	Size sizeText = getTextSize(dir, FONT_HERSHEY_DUPLEX, 3, 3, 0);
	int width = cvRound((sizeImg.width - sizeText.width) / 2);
	Point org = Point(width, 100);
	putText(img, dir, org, FONT_HERSHEY_DUPLEX, 3, Scalar(255, 255, 255), 3, LINE_AA);

	line(img, lane[0], lane[1], Scalar(0, 255, 255), 5, LINE_AA);
	line(img, lane[2], lane[3], Scalar(0, 255, 255), 5, LINE_AA);

	return img;
}
