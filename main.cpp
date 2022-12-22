#include <iostream>
#include <filesystem>
#include <math.h>
#include <chrono>
#include <time.h>
#include <thread>

#include <opencv2//opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/utils/trace.hpp>

#include "yolo7.h"
#include "QueueFPS.hpp"
#include "utils.h"
//download youtube videos:
//https://offeo.com/download/youtube-downloader/
//https://www.yotec.net/guide-how-to-install-opencv-cuda-on-windows/

using namespace std;
using namespace cv;
using namespace dnn;

static const std::string WinName = "Deep learning object detection in OpenCV";
using std::filesystem::exists;
QueueFPS<Mat> framesQueue;
bool process = true;
QueueFPS<Mat> processedFramesQueue;
QueueFPS<std::vector<Mat>> predictionsQueue;
Yolo7 yolov7;
Net net;
//Utils utils;

int inpWidth = 640;//parser.get<int>("width");
int inpHeight = 640;//parser.get<int>("height");
float scale = 1.0/ 255;// parser.get<float>("scale");
Scalar mean0 = Scalar(0, 0, 0);//parser.get<Scalar>("mean");
//Scalar mean0 = Scalar(104, 117, 123);
//Scalar mean0 = Scalar(114, 114, 114);
bool swapRB = true; //parser.get<bool>("rgb");
bool crop = false;
float confThreshold = 0.5f;// parser.get<float>("thr");
float nmsThreshold = 0.4f;//parser.get<float>("nms");
int backend = 5;// parser.get<int>("backend");
/*Choose one of computation backends : "
"0: automatically (by default), "
"1: Halide language (http://halide-lang.org/), "
"2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
"3: OpenCV implementation, "
"4: VKCOM, "
"5: CUDA "*/
int target = 6;// parser.get<int>("target")
/*Choose one of target computation devices : "
"0: CPU target (by default), "
"1: OpenCL, "
"2: OpenCL fp16 (half-float precision), "
"3: VPU, "
"4: Vulkan, "
"6: CUDA, "
"7: CUDA fp16 (half-float preprocess) */

std::vector<std::string> classes;

//----------------------------------------------------
// Frames capturing thread
std::thread frameThread(VideoCapture& cap)
{
	cout << "Frames capturing thread" << endl;		
	std::thread framesThread([&]() {
		Mat frame;
		int ee = 0;
		while (process)
		{
			ee++;
			if (cap.grab()) {
				cap >> frame;
				if (!frame.empty())
				{
					framesQueue.push(frame.clone());
					//	cout << "frame:" << ee << endl;
					std::this_thread::sleep_for(100ms);
				}
				else
				{
					cerr << "END or ERROR:video frame empty!" << endl;
					//system("pause");
					break;
				}
			}
			else {
				cout << "ERROR: bad grab" << endl;
				std::this_thread::sleep_for(100ms);				
			}
		}
	});
	return framesThread;
}

//----------------------------------------------------
// Frames processing thread
std::thread processingThread(Yolo7& yolov7_a)
{
	cout << "Processing thread" << endl;
	std::thread prThread([&]() {
		std::queue<AsyncArray> futureOutputs;
		Mat blob;	
		vector<Output> result;
		while (process)
		{
			// Get a next frame
			Mat frame;
			{
				if (!framesQueue.empty())
				{
					frame = framesQueue.get();
					framesQueue.clear();  // Skip the rest of frames
				}
			}
			// Process the frame
			if (!frame.empty())
			{
				yolov7_a.Preprocess(frame, net, Size(inpWidth, inpHeight), scale, mean0, swapRB, crop);
				std::vector<Mat> outs;
				net.forward(outs, net.getUnconnectedOutLayersNames());
				predictionsQueue.push(outs);
				processedFramesQueue.push(frame);

			}
		}
	});
	return prThread;
}

static void help(int argc, const char** argv)
{

	for (int ii = 1; ii < argc; ii++) {
		cout << argv[ii] << endl;
	}
}

//----------------------------------------------------
// MAIN
//----------------------------------------------------
const char* keys =
{
	"{help h ?| | show help message}{model|| <x.onnx>}{class|models/coco_classes.txt| <*.txt>}{image|| <*.png,jpg,bmp>}{video|| <*.mp4>}{path|.| path to file}{wr|0|writing to file}{gpu|1| Default gpu }"
};

int main(int argc, const char** argv)
{	
	/* Examples:
		"ImageDetector-yolov7-opencv.exe -h"
		"ImageDetector-yolov7-opencv.exe -model=models/yolov7-tiny.onnx -image=images/bus.jpg"
		"ImageDetector-yolov7-opencv.exe -model=models/yolov7-tiny.onnx -video=images/images/cat.mp4"
	*/

	cv::CommandLineParser parser(argc, argv, keys);

	parser.about("Trying OpenCV commandline parser");
	help(argc, argv);
	if (parser.has("help"))
	{		
		parser.printMessage();
		//parser.printErrors();
		return 0;
	}

	string path_name = parser.get<string>("path");
	if (path_name == ".") {
		path_name = std::filesystem::current_path().string();
	}

	string model_name = parser.get<string>("model");
	string model_path = path_name + "/" + model_name;
	if (!exists(model_path)) {
		cout << "ERROR: model file not exist" << endl;
		return 1;
	}

	string class_name = parser.get<string>("class");
	string class_path = path_name + "/" + class_name;
	if (!exists(class_path)) {
		cout << "ERROR: class file not exist" << endl;
		return 1;
	}
	cout << "class:" << model_name << endl;	
	string image_name = parser.get<string>("image");
	string video_name = parser.get<string>("video");

	string img_path = "";
	string video_path = "";
	bool image = false;
	bool mp4 = false;
	if (!image_name.empty()) {
		img_path = path_name + "/" + image_name;
		string ext = std::filesystem::path(img_path).extension().string();
		if ((ext == ".png") || (ext == ".jpg") || (ext == ".bmp")) {
			image = true;
		}
		else {
			cout << "image ext. is not png or jpg or bmp" << video_name << endl;
			return 1;
		}
		if (!exists(img_path)) {
			cout << "ERROR: image file not exist" << endl;
			return 1;
		}
		cout << "image:" << image_name << endl;
	}
	else 
	if (!video_name.empty()) {
		video_path = path_name + "/" + video_name;
		string ext = std::filesystem::path(video_path).extension().string();
		if (ext == ".mp4") {
			mp4 = true;
		}
		else {
			cout << "video ext. is not mp4" << video_name << endl;
			return 1;
		}
		if (!exists(video_path)) {
			cout << "ERROR: video file not exist" << endl;
			return 1;
		}		
		cout << "video:" << video_name << endl;
	}
	else {
		cout << "ERROR:image name or video name is empty" << endl;
		return 1;
	}

	bool wr = (bool)parser.get<int>("wr");
	bool gpu = (bool)parser.get<int>("gpu");
	if (!gpu) {
		backend = 0;
		target = 0;
	}

	Utils utils;
	
	string config_path = ""; // error fp16, need fp32
	
	if (gpu) {
		if (!utils.IsCUDA())
		{
			cout << "not founded GPU or/and CUDA" << endl;
			return -1;
		}
	}
	// python export.py --weights yolov7-tiny.pt --end2end --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
	// not --grid argument	
	

	std::vector<std::string> class_names = utils.LoadNames(class_path);//read classes
	if (class_names.empty()) {
		cout << "LoadNames is failed:" << class_path << endl;
		return false;
	}	
	if (yolov7.readModel(net, model_path, class_names, config_path, gpu)) {
		cout << "read net ok!" << endl;
		net.setPreferableBackend(backend);
		net.setPreferableTarget(target);
	}
	else {
		cout << "read onnx model failed!";
		return -1;
	}

	namedWindow(WinName, WINDOW_NORMAL);
	VideoCapture cap;
	VideoWriter video_raw;

	//generate random colors
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}

	if (image)
	{ // one image
		vector<Output> result;
		// Open an image file.
		Mat img = imread(img_path);
		utils.Timer(true);

		yolov7.Preprocess(img, net, Size(inpWidth, inpHeight), scale, mean0, swapRB, crop);
		std::vector<Mat> outs;
		net.forward(outs, net.getUnconnectedOutLayersNames());		
		if (yolov7.Postprocess(img, outs, result, net)) {
			yolov7.drawPred(img, result, color);
			cout << "Done:" << utils.Timer(false) << "ms" << endl;
			if (wr) {				
				string filename = std::filesystem::path(img_path).stem().string();
				imwrite(filename + "_o.png", img); // result image to file.
			}
			imshow("result image", img);			
			waitKey();
		}
		else {
			cout << "Detect Failed!" << endl;
			return 1;
		}		
	} else 
	if (mp4) {
		// Open a video file or an image file.		
		cap.open(video_path);
		if (!cap.isOpened()) {
			cerr << "ERRON:Unable to open video" << endl;
			return -1;
		}		
		if (wr) {
			int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
			int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
			string filename = std::filesystem::path(video_path).stem().string();
			if (!video_raw.open(filename + "_o.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 10, Size(frame_width, frame_height))) {
				cout << "ERRON:VideoWriter opened failed!" << endl;
				return 1;
			}
		}
		std::thread frmThread = frameThread(cap);
		std::thread prThread = processingThread(yolov7);

		// Postprocessing and rendering loop
		while (waitKey(1) < 0)
		{
			if (predictionsQueue.empty())
				continue;			
			std::vector<Mat> outs = predictionsQueue.get();
			Mat frame = processedFramesQueue.get();

			vector<Output> result;
			if (yolov7.Postprocess(frame, outs, result, net)) {
				yolov7.drawPred(frame, result, color);
			}

			if (predictionsQueue.counter > 1)
			{
				std::string label = format("Camera: %.2f FPS", framesQueue.getFPS());
				putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0));
				label = format("Network: %.2f FPS", predictionsQueue.getFPS());
				putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0));				
			}
			if (wr) {
				if (!video_raw.isOpened()) {
					cout << "ERRON:VideoWriter opened failed!" << endl;
					return 1;
				}
				video_raw << frame;
			}
			imshow(WinName, frame);
		}
		process = false;
		frmThread.join();
		prThread.join();
	}	
	if (cap.isOpened()) {
		cap.release();
	}
	if (video_raw.isOpened()) {
		video_raw.release();
	}
	//system("pause");		
	return 0;
}