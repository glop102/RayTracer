#pragma once
#include <string>
#include <format>
#include <iostream>
#include <memory>
#include <cmath>
#include <random>
#include <chrono>

const double Infinity = std::numeric_limits<double>::infinity();
const double PI = 3.1415926535897932385;

#ifdef stl_random
extern std::random_device random_seed_device; // used for seeding
extern thread_local std::default_random_engine gen; // common generator to pass into distributions
// extern thread_local std::mt19937 gen;
// extern thread_local pcg32 gen;
extern std::uniform_real_distribution<double> random_percentage_distribution; // range from 0 - 1
extern std::uniform_real_distribution<double> random_neg_pos_one; // range from -1 - 1
#else
extern std::random_device random_seed_device; // used for seeding
extern double pcg();
extern double (*gen)();
extern double random_percentage_distribution(double(*seeder)()); // range from 0 - 1
extern double random_neg_pos_one(double(*seeder)()); // range from -1 - 1
extern double random_range(double(*seeder)(),const double min,const double max);
#endif

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