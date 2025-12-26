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

BBox HittableList::bbox()const{
    switch(objects.size()){
        case 0: return BBox{0};
        case 1: return objects[0]->bbox();
        default:
            BBox b = objects[0]->bbox();
            for(int x=1; x<objects.size(); x++){
                BBox ob = objects[x]->bbox();
                Vector3::min_accum(b.min,ob.min);
                Vector3::max_accum(b.max,ob.max);
            }
            return b;
    }
}


//===================================================================
// BVHList
//===================================================================
BVHList::BVHList(ObjList& world_objects,int max_depth)
: max_depth_allowed(max_depth), objects(world_objects) {
    if(max_depth_allowed<=0 || objects.size() <= 1){
        // Recursion end case of max depth or only a single object
        // this should be the only case where left or right are null
        right = left = nullptr;
        if(objects.size()>=1){
            memoized_bbox = objects[0]->bbox();
            for(int x=1;x<objects.size();x++){
                BBox ob = objects[x]->bbox();
                Vector3::min_accum(memoized_bbox.min,ob.min);
                Vector3::max_accum(memoized_bbox.max,ob.max);
            }
        } else {
            // Make sure the bounding box is zero size to try to never have it get hit when we have no objects to hold
            memoized_bbox.max = memoized_bbox.min = {0.0,0.0,0.0};
        }
    } else if(objects.size() == 2) {
        //efficency case of two objects so there is no decision making of how to split them
        memoized_bbox = objects[0]->bbox();
        BBox ob = objects[1]->bbox();
        Vector3::min_accum(memoized_bbox.min,ob.min);
        Vector3::max_accum(memoized_bbox.max,ob.max);
        ObjList leftsplit(objects.begin(),objects.begin()+1);
        ObjList rightsplit(objects.begin()+1,objects.end());
        left = new BVHList(leftsplit,max_depth_allowed-1);
        right = new BVHList(rightsplit,max_depth_allowed-1);
        objects.clear(); // no reason to hold onto the objects list since we will never check them
    } else {
        ObjList sorted(objects);
        std::sort(sorted.begin(), sorted.end(), [](const std::shared_ptr<Hittable>& obj1,const std::shared_ptr<Hittable>& obj2){return obj1->bbox().min.x < obj2->bbox().min.x;} );
        BBox xleft,xright;
        auto xslices = minimal_surface_area_split(sorted,xleft,xright);

        std::sort(sorted.begin(), sorted.end(), [](const std::shared_ptr<Hittable>& obj1,const std::shared_ptr<Hittable>& obj2){return obj1->bbox().min.y < obj2->bbox().min.y;} );
        BBox yleft,yright;
        auto yslices = minimal_surface_area_split(sorted,yleft,yright);
        
        std::sort(sorted.begin(), sorted.end(), [](const std::shared_ptr<Hittable>& obj1,const std::shared_ptr<Hittable>& obj2){return obj1->bbox().min.z < obj2->bbox().min.z;} );
        BBox zleft,zright;
        auto zslices = minimal_surface_area_split(sorted,zleft,zright);

        //we have our slices - so lets find the smallest
        double xsize = xleft.half_surface_area() + xright.half_surface_area();
        double ysize = yleft.half_surface_area() + yright.half_surface_area();
        double zsize = zleft.half_surface_area() + zright.half_surface_area();
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
        Vector3::min_accum(xleft.min,xright.min);
        Vector3::max_accum(xleft.max,xright.max);
        memoized_bbox = xleft;
        objects.clear(); // no reason to hold onto the objects list since we will never check them
    }
}

BVHList::~BVHList(){
    if(left) delete left;
    if(right) delete right;
}

BBox BVHList::bbox()const{
    return memoized_bbox;
}

std::pair<BVHList::ObjList, BVHList::ObjList> BVHList::minimal_surface_area_split(ObjList& dividing_objects, BBox& left, BBox& right){
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
    left = dividing_objects[leftidx]->bbox();
    right = dividing_objects[rightidx]->bbox();

    while(leftidx+1 < rightidx){
        double left_bb_size = left.half_surface_area();
        double right_bb_size = right.half_surface_area();
        if(left_bb_size < right_bb_size){
            leftidx ++;
            auto obj_bbox = dividing_objects[leftidx]->bbox();
            Vector3::min_accum(left.min,obj_bbox.min);
            Vector3::max_accum(left.max,obj_bbox.max);
        }else{
            rightidx --;
            auto obj_bbox = dividing_objects[rightidx]->bbox();
            Vector3::min_accum(right.min,obj_bbox.min);
            Vector3::max_accum(right.max,obj_bbox.max);
        }
    }
    auto rightBegin = dividing_objects.begin()+rightidx;
    return std::make_pair<ObjList,ObjList>(std::vector(dividing_objects.begin(),rightBegin),std::vector(rightBegin,dividing_objects.end()));
}


bool BVHList::hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const{
    auto t = memoized_bbox.intersection_distance(ray);

    if(t.max >= t.min && t.max > 0.0 && t.max > allowed_distance.min && t.min < allowed_distance.max){
        // The ray intersects our box or originates inside our box
        return hit_internal(ray,allowed_distance,rec);
    }else{
        //missed the box
        return false;
    }
}

bool BVHList::hit_internal(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const{
    bool found_hit = false;

    // Check if we are a leaf node, so just a linear search through all the objects, likely only a few
    // We can assume that leaf nodes will have no neighbor in left or right, and non-leaf nodes will have
    // both left and right due to how the constructor works.
    if (!left) {
        // copied the HittableList function
        for(int x = 0; x < objects.size(); x++){
            found_hit |= objects[x]->hit(ray,allowed_distance,rec);
        }
        return found_hit;
    }

    // Not a leaf node, determine which child box to recurse down into until we find a leaf
    auto tleft  = left->memoized_bbox.intersection_distance(ray);
    auto tright = right->memoized_bbox.intersection_distance(ray);

    // test the closer box first to try to find a closer hit first and so we can skip testing the other box
    if(tleft.max < tright.max){
        if( tleft.min <= tleft.max && tleft.max >= allowed_distance.min && tleft.min <= allowed_distance.max ){
            found_hit |= left->hit_internal(ray,allowed_distance,rec);
        }
        if( tright.min <= tright.max && tright.max >= allowed_distance.min && tright.min <= allowed_distance.max ){
            found_hit |= right->hit_internal(ray,allowed_distance,rec);
        }
    }else{
        if( tright.min <= tright.max && tright.max >= allowed_distance.min && tright.min <= allowed_distance.max ){
            found_hit |= right->hit_internal(ray,allowed_distance,rec);
        }
        if( tleft.min <= tleft.max && tleft.max >= allowed_distance.min && tleft.min <= allowed_distance.max ){
            found_hit |= left->hit_internal(ray,allowed_distance,rec);
        }
    }

    return found_hit;
}