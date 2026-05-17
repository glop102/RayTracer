#include "utils.h"
#include "time.h"

std::random_device random_seed_device;


RealRange::RealRange():min(Infinity),max(-Infinity){}
RealRange::RealRange(double min, double max):min(min),max(max){}
double RealRange::size()const{
    return max-min;
}
// x within range - includes end ranges as valid
bool RealRange::contains(double x)const{
    return min <= x && x <= max;
}
// x within range - excludes end ranges as valid
bool RealRange::surrounds(double x)const{
    return min < x && x < max;
}
double RealRange::clamp(double x)const{
    if (x<min) return min;
    if (x>max) return max;
    return x;
}
const RealRange RealRange::empty = RealRange(+Infinity,-Infinity);
const RealRange RealRange::universe = RealRange(-Infinity,+Infinity);


Stopwatch::Stopwatch():startTime(Clock::now()){}
void Stopwatch::reset(){
    startTime = Clock::now();
}
std::chrono::milliseconds Stopwatch::duration()const{
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - startTime);
}


std::string ms_to_human(std::chrono::milliseconds duration){
    unsigned int ms = duration.count() %1000;
    duration /=1000;
    if(duration.count()==0){
        return std::format("{}ms",ms);
    }
    unsigned int s = duration.count() %60;
    duration/=60;
    if(duration.count()==0){
        return std::format("{}.{:03d}s",s,ms);
    }
    unsigned int m = duration.count() %60;
    duration /=60;
    if(duration.count()==0){
        return std::format("{}m {}.{:03d}s",m,s,ms);
    }
    return std::format("{}h {}m {}.{:03d}s",duration.count(),m,s,ms);
}