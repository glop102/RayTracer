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
    } else {
        memoized_bbox = objects[0]->bbox();
        for(int i=1; i<objects.size(); i++){
            memoized_bbox.absorb(objects[i]->bbox());
        }

        // x-sort
        ObjList sorted(objects);
        std::sort(sorted.begin(), sorted.end(), [](const std::shared_ptr<Hittable>& obj1,const std::shared_ptr<Hittable>& obj2){return obj1->bbox().min.x < obj2->bbox().min.x;} );
        BBox xleft_bbox,xright_bbox;
        auto xslices = minimal_surface_area_split(sorted,xleft_bbox,xright_bbox);
        
        // y-sort
        std::sort(sorted.begin(), sorted.end(), [](const std::shared_ptr<Hittable>& obj1,const std::shared_ptr<Hittable>& obj2){return obj1->bbox().min.y < obj2->bbox().min.y;} );
        BBox yleft_bbox,yright_bbox;
        auto yslices = minimal_surface_area_split(sorted,yleft_bbox,yright_bbox);
        
        // z-sort
        std::sort(sorted.begin(), sorted.end(), [](const std::shared_ptr<Hittable>& obj1,const std::shared_ptr<Hittable>& obj2){return obj1->bbox().min.z < obj2->bbox().min.z;} );
        BBox zleft_bbox,zright_bbox;
        auto zslices = minimal_surface_area_split(sorted,zleft_bbox,zright_bbox);

        //we have our slices - so lets find the smallest
        double xsize = (xslices.first.size()/4.0) * xleft_bbox.half_surface_area() + (xslices.second.size()/4.0) * xright_bbox.half_surface_area();
        double ysize = (yslices.first.size()/4.0) * yleft_bbox.half_surface_area() + (yslices.second.size()/4.0) * yright_bbox.half_surface_area();
        double zsize = (zslices.first.size()/4.0) * zleft_bbox.half_surface_area() + (zslices.second.size()/4.0) * zright_bbox.half_surface_area();
        double smallest_size;
        std::pair<BVHList::ObjList,BVHList::ObjList>* smallest_slices;
        if(xsize < ysize && xsize < zsize){
            smallest_size = xsize;
            smallest_slices = &xslices;
        }else if(ysize < zsize){
            smallest_size = ysize;
            smallest_slices = &yslices;
        }else{
            smallest_size = zsize;
            smallest_slices = &zslices;
        }

        // Check if we have something that doesn't make sense to break up or if we should continue recursing down
        if (smallest_slices->first.size() == 0 || smallest_slices->second.size() == 0 || smallest_size >= (objects.size()/4.0) * memoized_bbox.half_surface_area()) {
            // No subdivision happened so we are a leaf
            // The objects are already assigned to our object so nothing left to do for it
            left = right = nullptr;
        } else {
            objects.clear(); // no reason to hold onto the objects list since we will never check them
            left = new BVHList(smallest_slices->first,max_depth-1);
            right = new BVHList(smallest_slices->second,max_depth-1);
        }
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
        if(dividing_objects.size())
            left = dividing_objects[0]->bbox();
        return std::make_pair<ObjList,ObjList>(ObjList(dividing_objects),ObjList());
    }
    //We assume that objects is already sorted in whatever way is sensible
    //So now we need to partition the list in half, and move that partition around until we find the minimal surface area
    //We want to try every single index as the splitting line and try to find one where the total surface area is minimized
    unsigned int best_split = 0;
    double best_split_area = std::numeric_limits<double>::max();
    BBox testing_bbox_left,testing_bbox_right;
    for( unsigned int split=1; split<dividing_objects.size(); split++ ){
        // Left split
        testing_bbox_left = dividing_objects[0]->bbox();
        for(int i=1; i<split; i++){
            testing_bbox_left.absorb(dividing_objects[i]->bbox());
        }
        // Right Split
        testing_bbox_right = dividing_objects[split]->bbox();
        for(int i=split+1; i<dividing_objects.size(); i++){
            testing_bbox_right.absorb(dividing_objects[i]->bbox());
        }

        // The cost of a particular split should take into account how badly it is not evenly splitting the objects in the scene
        // That is why we multiply the area cost by the number of objects still in that contained area
        double testing_area_cost = (split / 4.0) * testing_bbox_left.half_surface_area() + ((dividing_objects.size() - split)/4.0) * testing_bbox_right.half_surface_area();
        if (testing_area_cost < best_split_area) {
            best_split_area = testing_area_cost;
            best_split = split;
            left = testing_bbox_left;
            right = testing_bbox_right;
        }
    }

    auto rightBegin = dividing_objects.begin()+best_split;
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
    if(tleft.min < tright.min){
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