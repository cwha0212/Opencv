#include <iostream>
#include "opencv2/opencv.hpp"
#include "facedetector_plus.h"

const String model = "res10_300x300_ssd_iter_140000_fp16.caffemodel";
const String config = "deploy.prototxt";
int cnt = 1;

void FaceDetector_plus::MakeTrainFiles(){
	VideoCapture cap(0);

	if(!cap.isOpened()){
		cerr << "Camera open failed!" << endl;
	}

	Net net = readNet(model,config);

	if(net.empty()){
		cerr << "Net open failed!" << endl;
	}

	Mat frame;
	while(true){
		char train[50];
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
			if (confidence > 0.8){
				sprintf(train,"./train/train_%d.jpg", cnt);
				Mat save_img = frame(Rect(Point(x1,y1),Point(x2,y2)));
				imwrite(train, save_img);
				cnt++;
			}
			String label = format("Face: %4.3f", confidence);
			putText(frame, label, Point(x1,y1-1), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,0,255));
		}
		imshow("frame", frame);
		if (waitKey(1) == 27){
			break;
		}
		if (cnt > 50){
			break;
		}
	}
}

Ptr<KNearest> FaceDetector_plus::train_knn(){
	char read[50];
	Mat train_images, train_labels;
	for (int i=1; i<=50; i++){
		sprintf(read,"./train/train_%d.jpg",i);
		Mat face = imread(read,IMREAD_GRAYSCALE);

		Mat roi, roi_float, roi_flatten;
		resize(face, roi, Size(320,320));
		roi.convertTo(roi_float, CV_32F);
		roi_flatten = roi_float.reshape(1,1);

		train_images.push_back(roi_flatten);
		train_labels.push_back(i/50);
	}
	Ptr<KNearest> knn = KNearest::create();
	knn -> train(train_images, ROW_SAMPLE, train_labels);

	return knn;
}
