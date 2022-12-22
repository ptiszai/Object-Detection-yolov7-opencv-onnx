#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>


struct Output {
	int id;             //result category id
	float confidence;   //result confidence
	cv::Rect box;       //rectangle
};

class Yolo7 {
public:
		Yolo7(){};
		~Yolo7(){}
		bool readModel(cv::dnn::Net& net, std::string& netPath, std::vector<std::string>& class_names_a, std::string netPathConfig = "", bool isCuda = false);		
		bool Postprocess(cv::Mat& SrcImg, std::vector<cv::Mat>& netOutputImg, std::vector<Output>& output, cv::dnn::Net& net);
		void Preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale_a, const cv::Scalar& mean_a, bool swapRB, bool crop);
		void drawPred(cv::Mat& img, std::vector<Output> result, std::vector<cv::Scalar> color);
private:	
	float sigmoid_x(float x)
	{
		return static_cast<float>(1.f / (1.f + exp(-x)));
	}

	const float netAnchors[3][6] = { {12, 16, 19, 36, 40, 28},{36, 75, 76, 55, 72, 146},{142, 110, 192, 243, 459, 401} }; //yolov7-P5 anchors
	const int netWidth = 640;   //ONNX image input width
	const int netHeight = 640;  //ONNX picture input height
	const int strideSize = 3;   //stride size

	const float netStride[4] = { 8, 16.0,32,64 };

	float boxThreshold = 0.25;
	float classThreshold = 0.25;
	float nmsThreshold = 0.45;
	float nmsScoreThreshold = boxThreshold * classThreshold;
	std::vector<std::string> class_names;
};
