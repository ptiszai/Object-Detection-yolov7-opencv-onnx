#include "utils.h"
#include <string>
#include <fstream>
#include <iostream>

#include <opencv2//opencv.hpp>
#include <opencv2/core/cuda.hpp>
using namespace std;
using namespace cv;


bool Utils::IsCUDA()
{
	int gpucount = cuda::getCudaEnabledDeviceCount();
	if (gpucount != 0) {
		cout << "no. of gpu = " << gpucount << endl;
	}
	else
	{
		cout << "There is no CUDA supported GPU" << endl;
		return false;

	}
	cuda::DeviceInfo deviceinfo;
	int id = deviceinfo.cuda::DeviceInfo::deviceID();
	cuda::setDevice(id);
	cuda::resetDevice();
	//enum cuda::FeatureSet arch_avail;
	//if (cuda::TargetArchs::builtWith(arch_avail))
	//	cout << "yes, this Gpu arch is supported" << endl;

	//cuda::DeviceInfo deviceinfo;
	cout << "GPU: " << deviceinfo.cuda::DeviceInfo::name() << endl;
	return true;
}

std::vector<std::string> Utils::LoadNames(const std::string& path = "") {
	// load class names
	std::vector<std::string> class_names;
	//std::string Path = "SS";
	std::ifstream infile(path);
	if (infile.is_open()) {

		std::string line;
		while (std::getline(infile, line)) {

			class_names.emplace_back(line);
		}
		infile.close();
	}
	else {
		std::cerr << "Error loading the class names!\n";
	}
	return class_names;
}

std::string Utils::Timer(bool start)
{
	static std::chrono::high_resolution_clock::time_point t0;
	
	std::string result = "";
	if (start)
	{
		t0 = std::chrono::high_resolution_clock::now();		
	}
	else
	{ //stop
		auto t1 = std::chrono::high_resolution_clock::now();
		auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);		
		result = to_string(int_ms.count());
	}
	return result; // ms
}