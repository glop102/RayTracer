#include <vector>
#include <random>
#include "utils.h"
#include "camera.h"
#include "shapes.h"
#include "scene.h"

void populate_random_spheres_volume(HittableList& list, int num_spheres, RealRange radius_range, double dx, double dy, double dz, int glass_frequency=12){
    while(num_spheres){
        num_spheres--;
        double new_r = random_percentage_distribution(gen) * (radius_range.max - radius_range.min) + radius_range.min;
        list.add(std::make_shared<Sphere>(
            Vector3{dx*random_neg_pos_one(gen),dy*random_neg_pos_one(gen),dz*random_neg_pos_one(gen)},
            new_r,
            num_spheres%glass_frequency==0 ? 
                (std::shared_ptr<Material>) std::make_shared<PureTransparentMaterial>(PureTransparentMaterial(1.5)) :
                (std::shared_ptr<Material>) std::make_shared<BRDMaterial>(BRDMaterial::random())
            ));
    }
}
void populate_random_spheres_plane_sitting(HittableList& list, int num_spheres, RealRange radius_range, double dx, double dz){
    while(num_spheres){
        num_spheres--;
        double new_r = random_percentage_distribution(gen) * (radius_range.max - radius_range.min) + radius_range.min;
        list.add(std::make_shared<Sphere>(
            Vector3{dx*random_neg_pos_one(gen),new_r,dz*random_neg_pos_one(gen)},
            new_r,
            std::make_shared<BRDMaterial>(BRDMaterial::random())
            ));
    }
}

int main(){
    // Camera viewport(1920*4,1080*4);
    Camera viewport(1920,1080);
    // Camera viewport(1920/2,1080/2);
    // viewport.sampling_per_pixel = 1000;
    // viewport.ongoing_image_export = 32;
    HittableList spheres;
    // populate_random_spheres_plane_sitting(spheres,200,RealRange{0.5,4},50,50);
    // populate_random_spheres_volume(spheres,1000,RealRange{0.5,4},50,50,50);
    populate_random_spheres_volume(spheres,100,RealRange{0.5,4},20,20,20);
    // populate_random_spheres_volume(spheres,20,RealRange{0.5,4},10,10,10);

    Stopwatch timer,totalTimer;
    BVHList world(spheres.objects);
    print("BVH Creation Time  {}\n",timer.duration());
    //Horizontal Rotation
    int number_frames = 120;
    for(int frame=0; frame < number_frames; frame++){
        viewport.origin = Vector3{cos(2*PI*(frame/(double)number_frames))*25,sin(2*PI*(frame/(double)number_frames))*25,25};
        viewport.up_direction = y_pos;
        viewport.look_at(Vector3{0,0,0});
        timer.reset();
        // viewport.render(spheres);
        viewport.render(world);
        print("Frame: {} - {}\n",frame,ms_to_human(timer.duration()));
        viewport.threaded_write_to_png(std::format("video/{}.png",frame));
    }

    viewport.write_to_png("output_bvh.png");
    // timer.reset();
    // viewport.render(spheres);
    // print("Frame: {}\n",ms_to_human(timer.duration()));
    // viewport.write_to_png("output_sphere.png");
    // viewport.write_to_png("19200x10800_10Krays.png");
    print("\n\nTotal Time {}\n",ms_to_human(totalTimer.duration()));
    return 0;
}