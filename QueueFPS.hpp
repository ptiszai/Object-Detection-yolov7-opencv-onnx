#pragma once

#include <queue>
#include <mutex>
#include <opencv2/core/utility.hpp>

using namespace cv;

template <typename T> class QueueFPS : public std::queue<T>
{
public:
    QueueFPS() : counter(0) {}

    void push(const T& entry)
    {
        std::lock_guard<std::mutex> lock(mutex);

        std::queue<T>::push(entry);
        counter += 1;
        if (counter == 1)
        {
            // Start counting from a second frame (warmup).
            tm.reset();
            tm.start();
        }
    }

    T get()
    {
        std::lock_guard<std::mutex> lock(mutex);
        T entry = this->front();
        this->pop();
        return entry;
    }

    float getFPS()
    {
        tm.stop();
        double fps = counter / tm.getTimeSec();
        tm.start();
        return static_cast<float>(fps);
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex);
        while (!this->empty())
            this->pop();
    }

    unsigned int counter;

private:
    TickMeter tm;
    std::mutex mutex;
};

/*
class QueueFPS : public std::queue<T> {
public:
    QueueFPS() : counter(0) {}
private:
    unsigned int counter;
    TickMeter tm;
    std::mutex mutex;

    void push(const T& entry);
    T get();
    float getFPS();
    void clear();
};*/
