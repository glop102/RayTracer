#pragma once
#include <string>
#include <format>
#include <iostream>
#include <memory>
#include <cmath>
#include <chrono>
#include <random>

const double Infinity = std::numeric_limits<double>::infinity();
const double PI = 3.1415926535897932385;

extern std::random_device random_seed_device;

inline double rng() {
    thread_local unsigned int seed = random_seed_device();
    unsigned int state = seed * 747796405u + 2891336453u;
    unsigned int word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    seed = (word >> 22u) ^ word;
    return seed / (double)std::numeric_limits<unsigned int>::max();
}
inline double random_percentage_distribution() { return rng(); }
inline double random_neg_pos_one()              { return rng() * 2.0 - 1.0; }
inline double random_range(double min, double max) { return rng() * (max - min) + min; }

template<typename... Args>
void print(const char* fmnt, Args... args){
    auto s = std::vformat(
        fmnt,
        std::make_format_args(args...)
        );
    std::cout << s;
}

class RealRange{
    public:
    double min,max;
    RealRange();
    RealRange(double min, double max);
    double size()const; // max - min
    bool contains(double x)const; // x within range - includes end ranges as valid
    bool surrounds(double x)const; // x within range - excludes end ranges as valid
    double clamp(double x)const; // return a value that is clamped to within this range
    static const RealRange empty, universe;
};

class Stopwatch{
    typedef std::chrono::steady_clock Clock;
    protected:
    std::chrono::time_point<Clock> startTime;
    public:
    Stopwatch();
    void reset();
    std::chrono::milliseconds duration()const;
};

std::string ms_to_human(std::chrono::milliseconds duration);