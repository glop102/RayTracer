#include "utils.h"
#include "time.h"

#ifdef stl_random
std::random_device random_seed_device; // used for seeding
thread_local std::default_random_engine gen(random_seed_device());
// thread_local std::mt19937 gen(random_seed_device());
// thread_local pcg32 gen(random_seed_device());
std::uniform_real_distribution<double> random_percentage_distribution(0.0,1.0);
std::uniform_real_distribution<double> random_neg_pos_one(-1.0,1.0); // range from -1 - 1
#else
std::random_device random_seed_device; // used for seeding
double pcg(){
    thread_local static unsigned int seed = random_seed_device();
    unsigned int state = seed * 747796405 + 2891336453;
    unsigned int word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    seed = (word >> 22) ^ word;
    return seed / (double) std::numeric_limits<unsigned int>::max();
}
double (*gen)() = pcg;
double random_percentage_distribution(double(*seeder)()){
    return seeder();
}
double random_neg_pos_one(double(*seeder)()){
    return seeder()*2.0 - 1.0;
}
double random_range(double(*seeder)(),const double min,const double max){
    return seeder()*(max-min) + min;
}
#endif


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