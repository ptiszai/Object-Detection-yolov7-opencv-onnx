#pragma once
#include <string>
#include <vector>

class Utils {
public:
    Utils() {}
    ~Utils() {}
    std::vector<std::string> LoadNames(const std::string& path);
    bool IsCUDA();
    std::string Timer(bool start);
private:
};
