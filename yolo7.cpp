#include"yolo7.h"
#include <fstream>
#include <string>
using namespace std;
using namespace cv;
using namespace cv::dnn;

bool Yolo7::readModel(Net& net, string& netPath, std::vector<std::string>& class_names_a, string netPathConfig, bool isCuda) {
	try {		
		net = readNetFromONNX(netPath);
	}
	catch (const std::exception& exp) 
	{
		cout << "read onnx model failed:" << exp.what() << endl;
		return false;
	}
	//cuda
	if (isCuda) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	//cpu
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	class_names = class_names_a;
	return true;
}

//----------------------------------------------------
void Yolo7::Preprocess(const Mat& frame, Net& net, Size inpSize, float scale_a, const Scalar& mean_a, bool swapRB, bool crop)
{
	static Mat blob;
	// Create a 4D blob from a frame.
	if (inpSize.width <= 0) inpSize.width = frame.cols;
	if (inpSize.height <= 0) inpSize.height = frame.rows;
	blobFromImage(frame, blob, scale_a, inpSize, mean_a, swapRB, crop, CV_32F);

	// Run a model.
	//net.setInput(blob, "", scale, mean_a);
	net.setInput(blob);
}

bool Yolo7::Postprocess(Mat& SrcImg, vector<Mat>& netOutputImg, vector<Output>& output, Net& net) {
	Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	//Mat netInputImg = SrcImg.clone();
	try {
/*		if (maxLen > 1.2 * col || maxLen > 1.2 * row) {
			Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
			SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
			netInputImg = resizeImg;
		}*/
#if CV_VERSION_MAJOR==4&&CV_VERSION_MINOR==6
		std::sort(netOutputImg.begin(), netOutputImg.end(), [](Mat& A, Mat& B) {return A.size[2] > B.size[2]; });//opencv 4.6
#endif
		std::vector<int> classIds;//result id array
		std::vector<float> confidences;//as a result, each id corresponds to a confidence array
		std::vector<cv::Rect> boxes;//each id rectangle
		float ratio_h = (float)row / netHeight;
		float ratio_w = (float)col / netWidth;
		int net_width = class_names.size() + 5;  //The output network width is the number of categories + 5
		for (int stride = 0; stride < strideSize; stride++) {    //stride
			float* pdata = (float*)netOutputImg[stride].data;
			int grid_x = (int)(netWidth / netStride[stride]);
			int grid_y = (int)(netHeight / netStride[stride]);
			for (int anchor = 0; anchor < 3; anchor++) {	//anchors
				const float anchor_w = netAnchors[stride][anchor * 2];
				const float anchor_h = netAnchors[stride][anchor * 2 + 1];
				for (int i = 0; i < grid_y; i++) {
					for (int j = 0; j < grid_x; j++) {
						float box_score = sigmoid_x(pdata[4]); ;//get the probability that an object is contained in the box of each row
						if (box_score >= boxThreshold) {
							cv::Mat scores(1, class_names.size(), CV_32FC1, pdata + 5);
							Point classIdPoint;
							double max_class_socre;
							minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
							max_class_socre = sigmoid_x(max_class_socre);
							if (max_class_socre >= classThreshold) {
								float x = (sigmoid_x(pdata[0]) * 2.f - 0.5f + j) * netStride[stride];  //x
								float y = (sigmoid_x(pdata[1]) * 2.f - 0.5f + i) * netStride[stride];   //y
								float w = powf(sigmoid_x(pdata[2]) * 2.f, 2.f) * anchor_w;   //w
								float h = powf(sigmoid_x(pdata[3]) * 2.f, 2.f) * anchor_h;  //h
								int left = (int)(x - 0.5 * w) * ratio_w + 0.5;
								int top = (int)(y - 0.5 * h) * ratio_h + 0.5;
								classIds.push_back(classIdPoint.x);
								confidences.push_back(max_class_socre * box_score);
								boxes.push_back(Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
							}
						}
						pdata += net_width;//next line
					}
				}
			}
		}
		// Perform non - maximum suppression to remove redundant overlapping boxes with lower confidence(NMS)
		vector<int> nms_result;
		NMSBoxes(boxes, confidences, nmsScoreThreshold, nmsThreshold, nms_result);
		for (int i = 0; i < nms_result.size(); i++) {
			int idx = nms_result[i];
			Output result;
			result.id = classIds[idx];
			result.confidence = confidences[idx];
			result.box = boxes[idx];
			output.push_back(result);
		}
		if (output.size())
			return true;
		else
			return false;
	}
	catch (const std::exception& exp)
	{
		cout << "Yolo7::Detect:" << exp.what() << endl;
		return false;
	}
}

void Yolo7::drawPred(Mat& img, vector<Output> result, vector<Scalar> color) {
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		float number = 30.0f;
		char buffer[20];  // maximum expected length of the float		
		std::snprintf(buffer, 20, "%.2f", result[i].confidence);		
		string label = class_names[result[i].id] + ":" + std::string(buffer);		
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);		
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
}
