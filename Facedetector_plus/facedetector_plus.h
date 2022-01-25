#ifndef __FACE_DETECTOR_PLUS_H__
#define __FACE_DETECTOR_PLUS_H__

#include<iostream>
#include"opencv2/opencv.hpp"

using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace cv::ml;

class FaceDetector_plus {
	public:
		void MakeTrainFiles();
		Ptr<KNearest> train_knn();
};

#endif
