#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void detect_face();
void detect_eyes();

int main()
{
	detect_face();
	detect_eyes();

	return 0;
}

void detect_face()
{
	VideoCapture cap(0);

	while (true){
		Mat src;
		Mat frame;
		cap >> src;
		cap >> frame;

		if (src.empty()) {
			cerr << "Image load failed!" << endl;
			return;
		}
		imshow("frame", frame);

		CascadeClassifier classifier("haarcascade_frontalface_default.xml");

		if (classifier.empty()) {
			cerr << "XML load failed!" << endl;
			return;
		}

		vector<Rect> faces;
		classifier.detectMultiScale(src, faces);

		for (Rect rc : faces) {
			rectangle(src, rc, Scalar(255, 0, 255), 2);
		}

		imshow("src", src);
		imshow("frame", frame);
		if(waitKey(10)==27){
			break;
		}
	}
	destroyAllWindows();
}

void detect_eyes()
{
	VideoCapture cap(0);

	while (true){
		Mat src;
		cap >> src;

		if (src.empty()) {
			cerr << "Image load failed!" << endl;
			return;
		}

		CascadeClassifier face_classifier("haarcascade_frontalface_default.xml");
		CascadeClassifier eye_classifier("haarcascade_eye.xml");

		if (face_classifier.empty() || eye_classifier.empty()) {
			cerr << "XML load failed!" << endl;
			return;
		}

		vector<Rect> faces;
		face_classifier.detectMultiScale(src, faces);

		for (Rect face : faces) {
			rectangle(src, face, Scalar(255, 0, 255), 2);

			Mat faceROI = src(face);
			vector<Rect> eyes;
			eye_classifier.detectMultiScale(faceROI, eyes);

			for (Rect eye : eyes) {
				Point center(eye.x + eye.width / 2, eye.y + eye.height / 2);
				circle(faceROI, center, eye.width / 2, Scalar(255, 0, 0), 2, LINE_AA);
			}
		}

		imshow("src", src);

		if(waitKey(10)==27){
			break;
		}
	}
	destroyAllWindows();
}