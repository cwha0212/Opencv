#include<iostream>
#include"opencv2/opencv.hpp"
#include"facedetector_plus.h"

const String model = "res10_300x300_ssd_iter_140000_fp16.caffemodel";
const String config = "deploy.prototxt";

int main(){
	Net net = readNet(model,config);
	FaceDetector_plus fdc;
	fdc.MakeTrainFiles();
	Ptr<KNearest> knn = fdc.train_knn();
	VideoCapture cap(0);
	while(true){
		int x1, x2, y1,y2;
		Mat frame;
		cap >> frame;
		if(frame.empty()){
			break;
		}

		Mat blob = blobFromImage(frame, 1, Size(300,300), Scalar(104,177,123));
		net.setInput(blob);
		Mat res = net.forward();

		Mat detect(res.size[2], res.size[3], CV_32FC1, res.ptr<float>());

		for(int i = 0; i < detect.rows; i++){
			float confidence = detect.at<float>(i,2);
			if(confidence < 0.5){
				break;
			}

			int x1 = cvRound(detect.at<float>(i,3) * frame.cols);
			int y1 = cvRound(detect.at<float>(i,4) * frame.rows);
			int x2 = cvRound(detect.at<float>(i,5) * frame.cols);
			int y2 = cvRound(detect.at<float>(i,6) * frame.rows);
			rectangle(frame,Rect(Point(x1,y1), Point(x2,y2)), Scalar(0,0,255));
		
			Mat img_resize, img_float, img_flatten, result;
			Mat img = frame(Rect(Point(x1,y1),Point(x2,y2)));
			resize(img, img_resize, Size(320, 320));
			img_resize.convertTo(img_float, CV_32F);
			img_flatten = img_float.reshape(1, 1);

			knn->findNearest(img_flatten,3,result);
			cout << cvRound(result.at<float>(0, 0)) << endl;	
		}

		imshow("frame", frame);
		if (waitKey(1) == 27){
			break;
		}
	}
}

