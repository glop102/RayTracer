#include "scene.h"
#include <utility>

using std::shared_ptr;
using std::make_shared;

bool HittableList::hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const{
    bool found_hit = false;
    // for(const auto& object : objects){
    //     if(object->hit(ray,allowed_distance,rec)){
    //         found_hit = true;
    //     }
    // }
    for(int x=0; x<objects.size(); x++){
        found_hit |= objects[x]->hit(ray,allowed_distance,rec);
    }
    return found_hit;
}

void HittableList::add(shared_ptr<Hittable> object){
    objects.push_back(object);
}
void HittableList::clear(){
    objects.clear();
}

Vector3 HittableList::min()const{
    switch(objects.size()){
        case 0: return Vector3{0,0,0};
        case 1: return objects[0]->min();
        default:
            Vector3 v = objects[0]->min();
            for(int x=1; x<objects.size(); x++){
                Vector3 om = objects[x]->min();
                if(om.x < v.x) v.x=om.x;
                if(om.y < v.y) v.y=om.y;
                if(om.z < v.z) v.z=om.z;
            }
            return v;
    }
}
Vector3 HittableList::max()const{
    switch(objects.size()){
        case 0: return Vector3{0,0,0};
        case 1: return objects[0]->max();
        default:
            Vector3 v = objects[0]->max();
            for(int x=1; x<objects.size(); x++){
                Vector3 om = objects[x]->max();
                if(om.x > v.x) v.x=om.x;
                if(om.y > v.y) v.y=om.y;
                if(om.z > v.z) v.z=om.z;
            }
            return v;
    }
}


//===================================================================
// BVHList
//===================================================================
BVHList::BVHList(ObjList& world_objects,int max_depth)
: max_depth_allowed(max_depth), objects(world_objects) {
    if(max_depth_allowed<=0 || objects.size() <= 1){
        right = left = nullptr;
        if(objects.size()>=1){
            memoized_min = objects[0]->min();
            memoized_max = objects[0]->max();
            for(int x=1;x<objects.size();x++){
                Vector3::min_accum(memoized_min,objects[x]->min());
                Vector3::max_accum(memoized_max,objects[x]->max());
            }
        } // else 0, so who cares what the memoized max/min is
    } else if(objects.size() == 2) {
        //efficency case of two objects so there is no decision making of how to split them
        memoized_min = objects[0]->min();
        memoized_max = objects[0]->max();
        Vector3::min_accum(memoized_min,objects[1]->min());
        Vector3::max_accum(memoized_max,objects[1]->max());
        ObjList leftsplit(objects.begin(),objects.begin()+1);
        ObjList rightsplit(objects.begin()+1,objects.end());
        left = new BVHList(leftsplit,max_depth_allowed-1);
        right = new BVHList(rightsplit,max_depth_allowed-1);
        objects.clear(); // no reason to hold onto the objects list since we will never check them
    } else {
        ObjList sorted(objects);
        std::sort(sorted.begin(), sorted.end(), [](const std::shared_ptr<Hittable>& obj1,const std::shared_ptr<Hittable>& obj2){return obj1->min().x < obj2->min().x;} );
        Vector3 xleftmin,xleftmax,xrightmin,xrightmax;
        auto xslices = minimal_surface_area_split(sorted,xleftmin,xleftmax,xrightmin,xrightmax);

        std::sort(sorted.begin(), sorted.end(), [](const std::shared_ptr<Hittable>& obj1,const std::shared_ptr<Hittable>& obj2){return obj1->min().y < obj2->min().y;} );
        Vector3 yleftmin,yleftmax,yrightmin,yrightmax;
        auto yslices = minimal_surface_area_split(sorted,yleftmin,yleftmax,yrightmin,yrightmax);
        
        std::sort(sorted.begin(), sorted.end(), [](const std::shared_ptr<Hittable>& obj1,const std::shared_ptr<Hittable>& obj2){return obj1->min().z < obj2->min().z;} );
        Vector3 zleftmin,zleftmax,zrightmin,zrightmax;
        auto zslices = minimal_surface_area_split(sorted,zleftmin,zleftmax,zrightmin,zrightmax);

        //we have our slices - so lets find the smallest
        double xsize = calc_bb_half_surface_area(xleftmin,xleftmax) + calc_bb_half_surface_area(xrightmin,xrightmax);
        double ysize = calc_bb_half_surface_area(yleftmin,yleftmax) + calc_bb_half_surface_area(yrightmin,yrightmax);
        double zsize = calc_bb_half_surface_area(zleftmin,zleftmax) + calc_bb_half_surface_area(zrightmin,zrightmax);
        if(xsize < ysize && xsize < zsize){
            left = new BVHList(xslices.first,max_depth_allowed-1);
            right = new BVHList(xslices.second,max_depth_allowed-1);
        }else if(ysize < zsize){
            left = new BVHList(yslices.first,max_depth_allowed-1);
            right = new BVHList(yslices.second,max_depth_allowed-1);
        }else{
            left = new BVHList(zslices.first,max_depth_allowed-1);
            right = new BVHList(zslices.second,max_depth_allowed-1);
        }
        // i should be able to just use the xmin/max for all 3 branches since it describes the totallity of our objects?
        Vector3::min_accum(xleftmin,xrightmin);
        Vector3::max_accum(xleftmax,xrightmax);
        memoized_min = xleftmin;
        memoized_max = xleftmax;
        objects.clear(); // no reason to hold onto the objects list since we will never check them
    }
}

BVHList::~BVHList(){
    if(left) delete left;
    if(right) delete right;
}

Vector3 BVHList::min()const{
    return memoized_min;
}
Vector3 BVHList::max()const{
    return memoized_max;
}

double BVHList::calc_bb_half_surface_area(const Vector3& min, const Vector3& max){
    Vector3 dv = max-min;
    return (dv.x*(dv.z+dv.y))+(dv.z*dv.y);
}

std::pair<BVHList::ObjList, BVHList::ObjList> BVHList::minimal_surface_area_split(ObjList& dividing_objects, Vector3& leftmin, Vector3& leftmax, Vector3& rightmin, Vector3& rightmax){
    //We are guarenteed two items in the objects list because the constructor is handling there being 0 or 1 items
    //But lets check anyways
    if(dividing_objects.size()<=1){
        return std::make_pair<ObjList,ObjList>(ObjList(dividing_objects),ObjList());
    }
    //We assume that objects is already sorted in whatever way is sensible
    //So now we need to partition the list in half, and move that partition around until we find the minimal surface area
    //Start by only taking in one element for each side and slowly accumulate objects
    unsigned int leftidx = 0;
    unsigned int rightidx = dividing_objects.size()-1;
    leftmin = dividing_objects[leftidx]->min();
    leftmax = dividing_objects[leftidx]->max();
    rightmin = dividing_objects[rightidx]->min();
    rightmax = dividing_objects[rightidx]->max();

    while(leftidx+1 < rightidx){
        double left_bb_size = calc_bb_half_surface_area(leftmin,leftmax);
        double right_bb_size = calc_bb_half_surface_area(rightmin,rightmax);
        if(left_bb_size < right_bb_size){
            leftidx ++;
            Vector3::min_accum(leftmin,dividing_objects[leftidx]->min());
            Vector3::max_accum(leftmax,dividing_objects[leftidx]->max());
        }else{
            rightidx --;
            Vector3::min_accum(rightmin,dividing_objects[rightidx]->min());
            Vector3::max_accum(rightmax,dividing_objects[rightidx]->max());
        }
    }
    auto rightBegin = dividing_objects.begin()+rightidx;
    return std::make_pair<ObjList,ObjList>(std::vector(dividing_objects.begin(),rightBegin),std::vector(rightBegin,dividing_objects.end()));
}


bool BVHList::hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const{
    auto [tmin,tmax] = determine_ray_intersections(ray);

    if(tmax >= tmin && tmax > 0.0){
        // The ray intersects our box
        return hit_internal(ray,allowed_distance,rec);
    }else{
        //missed the box
        return false;
    }
}

std::pair<double,double> BVHList::determine_ray_intersections(const Ray& ray)const{
    // Taken from https://tavianator.com/2011/ray_box.html
    double t1 = (memoized_min.x - ray.origin.x)/ray.direction.x;
    double t2 = (memoized_max.x - ray.origin.x)/ray.direction.x;
    double tmin = std::min(t1, t2);
    double tmax = std::max(t1, t2);

    t1 = (memoized_min.y - ray.origin.y)/ray.direction.y;
    t2 = (memoized_max.y - ray.origin.y)/ray.direction.y;
    tmin = std::max(tmin,std::min(t1, t2));
    tmax = std::min(tmax,std::max(t1, t2));

    t1 = (memoized_min.z - ray.origin.z)/ray.direction.z;
    t2 = (memoized_max.z - ray.origin.z)/ray.direction.z;
    tmin = std::max(tmin,std::min(t1, t2));
    tmax = std::min(tmax,std::max(t1, t2));
    return std::make_pair(tmin,tmax);
}

bool BVHList::hit_internal(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const{
    // Check if we are a leaf node, so just a linear search through all the objects, likely only a few
    bool found_hit = false;
    if(!(left&&right)){
        // copied the HittableList function
        for(int x = 0; x < objects.size(); x++){
            found_hit |= objects[x]->hit(ray,allowed_distance,rec);
        }
        return found_hit;
    }

    // Not a leaf node, determine which child box to recurse down into until we find a leaf
    auto [tminleft,tmaxleft] = left->determine_ray_intersections(ray);
    auto [tminright,tmaxright] = right->determine_ray_intersections(ray);

    if(tmaxleft < tmaxright){ // test the closer box first to try to find a closer hit first and so we can skip testing the other box
        if( tminleft <= tmaxleft && tmaxleft >= allowed_distance.min && tminleft <= allowed_distance.max ){
            found_hit |= left->hit_internal(ray,allowed_distance,rec);
        }
        if( tminright <= tmaxright && tmaxright >= allowed_distance.min && tminright <= allowed_distance.max ){
            found_hit |= right->hit_internal(ray,allowed_distance,rec);
        }
    }else{
        if( tminright <= tmaxright && tmaxright >= allowed_distance.min && tminright <= allowed_distance.max ){
            found_hit |= right->hit_internal(ray,allowed_distance,rec);
        }
        if( tminleft <= tmaxleft && tmaxleft >= allowed_distance.min && tminleft <= allowed_distance.max ){
            found_hit |= left->hit_internal(ray,allowed_distance,rec);
        }
    }

    return found_hit;
}