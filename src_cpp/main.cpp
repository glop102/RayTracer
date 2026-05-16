#include <vector>
#include <random>
#include <cstdlib>
#include <cstring>
#include "utils.h"
#include "camera.h"
#include "shapes.h"
#include "scene.h"
#include "model.h"

struct RenderConfig {
    int width = 1920;
    int height = 1080;
    int rays_per_pixel = 100;
    int ray_depth = 10;
};

static void print_usage(const char* prog) {
    print("Usage: {} [options]\n", prog);
    print("  --width  N       Image width  (default 1920)\n");
    print("  --height N       Image height (default 1080)\n");
    print("  --rays   N       Rays per pixel (default 100)\n");
    print("  --depth  N       Max ray bounce depth (default 10)\n");
    print("  --help           Show this message\n");
}

static RenderConfig parse_args(int argc, char** argv) {
    RenderConfig cfg;
    for (int i = 1; i < argc; ++i) {
        auto need_next = [&]() -> int {
            if (i + 1 >= argc) {
                print("Error: {} requires a value\n", argv[i]);
                std::exit(1);
            }
            return std::atoi(argv[++i]);
        };
        if (std::strcmp(argv[i], "--width")  == 0) cfg.width          = need_next();
        else if (std::strcmp(argv[i], "--height") == 0) cfg.height    = need_next();
        else if (std::strcmp(argv[i], "--rays")   == 0) cfg.rays_per_pixel = need_next();
        else if (std::strcmp(argv[i], "--depth")  == 0) cfg.ray_depth  = need_next();
        else if (std::strcmp(argv[i], "--help")   == 0) { print_usage(argv[0]); std::exit(0); }
        else { print("Unknown argument: {}\n", argv[i]); print_usage(argv[0]); std::exit(1); }
    }
    return cfg;
}

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

void populate_random_sphere_of_spheres(HittableList& list, int num_spheres, RealRange radius_range, double major_sphere_radius, int glass_frequency=12){
    auto glass = std::make_shared<PureTransparentMaterial>(PureTransparentMaterial(1.5));
    while(num_spheres){
        num_spheres--;
        double new_r = random_percentage_distribution(gen) * (radius_range.max - radius_range.min) + radius_range.min;
        list.add(std::make_shared<Sphere>(
            Vector3::random_unit_vector() * major_sphere_radius,
            new_r,
            num_spheres%glass_frequency==0 ? 
                (std::shared_ptr<Material>) glass :
                (std::shared_ptr<Material>) std::make_shared<BRDMaterial>(BRDMaterial::random())
            ));
    }
}

void populate_triangles_crafted_test(HittableList& list){
    auto glass = std::make_shared<PureTransparentMaterial>(1.5);

    // for ( auto& cube_tri : make_cube( 3.0, Point3{6.0,0.0,0.0}, AluminiumDull) ) {
    //     list.add(cube_tri);
    // }
    // for ( auto& cube_tri : make_cube( 3.0, Point3{0.0,12.0,0.0}, MetalShiny) ) {
    //     list.add(cube_tri);
    // }
    // for ( auto& cube_tri : make_cube( 3.0, Point3{0.0,0.0,6.0}, glass) ) {
    //     list.add(cube_tri);
    // }

    // load_ply_file("bunny/reconstruction/bun_zipper.ply", list, glass, 100.0, Point3 {0,-5.0,0});
    load_ply_file("bunny/reconstruction/bun_zipper.ply", list, AluminiumDull, 100.0, Point3 {0,-5.0,0});
}
void populate_sphere_crafted_test(HittableList& list){
    // "Horizon"
    list.add(std::make_shared<Sphere>(
        Vector3{0.0,-100.0,0.0},
        100.0,
        std::make_shared<BRDMaterial>(DarkGreen,White,Black,1.0,1.0)
    ));
    // Center - Basic
    list.add(std::make_shared<Sphere>(
        Vector3{0.0,4.0,0.0},
        4.0,
        std::make_shared<BRDMaterial>(DarkBlue,White,Black,1.0,1.0)
    ));
    // Right - Metal
    list.add(std::make_shared<Sphere>(
        Vector3{8.0,4.0,0.0},
        4.0,
        std::make_shared<BRDMaterial>(DarkBlue,White,Black,1.0,0.1)
    ));
    // Left - Glass
    list.add(std::make_shared<Sphere>(
        Vector3{-8.0,4.0,0.0},
        4.0,
        std::make_shared<PureTransparentMaterial>(1.5)
    ));
    // Left - Glass hollow
    list.add(std::make_shared<Sphere>(
        Vector3{-8.0,4.0,0.0},
        3.0,
        std::make_shared<PureTransparentMaterial>(1.0/1.5)
    ));
    // Left - Glowing Light
    // list.add(std::make_shared<Sphere>(
    //     Vector3{-8.0,6.0,4.0},
    //     0.5,
    //     std::make_shared<BRDMaterial>(Black,Black,White,1.0,0.1)
    // ));
}
void populate_hand_crafted_box_plus_embedded_sphere(HittableList& list){
    auto glass = std::make_shared<PureTransparentMaterial>(1.5);
    for ( auto& cube_tri : make_cube( 8.0, Point3{0.0,0.0,0.0}, glass) ) {
        list.add(cube_tri);
    }
    list.add(std::make_shared<Sphere>(
        Vector3{0.0,0.0,0.0},
        6.5,
        std::make_shared<BRDMaterial>(DarkBlue,White,Black,1.0,0.1)
    ));
}

int main(int argc, char** argv){
    RenderConfig cfg = parse_args(argc, argv);
    print("Config: {}x{}, {} rays/pixel, depth {}\n", cfg.width, cfg.height, cfg.rays_per_pixel, cfg.ray_depth);

    // Camera viewport(1920*4,1080*4);
    Camera viewport(cfg.width, cfg.height);
    // Camera viewport(1920/2,1080/2);
    viewport.sampling_per_pixel = cfg.rays_per_pixel;
    viewport.max_trace_depth    = cfg.ray_depth;
    // viewport.sampling_per_pixel = 1000;
    // viewport.ongoing_image_export = 32;
    HittableList spheres;
    // populate_random_spheres_plane_sitting(spheres,200,RealRange{0.5,4},50,50);
    // populate_random_spheres_volume(spheres,1000,RealRange{0.5,4},50,50,50);
    // populate_random_spheres_volume(spheres,100,RealRange{0.5,4},20,20,20);
    // populate_random_spheres_volume(spheres,20,RealRange{0.5,4},10,10,10);

    // populate_sphere_crafted_test(spheres);
    populate_triangles_crafted_test(spheres);
    // populate_hand_crafted_box_plus_embedded_sphere(spheres);
    populate_random_sphere_of_spheres(spheres,500,RealRange{2.0,6.0},100);

    Stopwatch timer,totalTimer;
    BVHList world(spheres.objects);
    print("BVH Creation Time  {}\n",timer.duration());
    world.debug_print_tree();

    //Horizontal Rotation
    int number_frames = 16;
    // for(int frame=0; frame < number_frames; frame++){
        int frame = 4; //58;
        viewport.origin = Vector3{cos(2*PI*(frame/(double)number_frames))*15,5,sin(2*PI*(frame/(double)number_frames))*15};
        viewport.look_at(Vector3{0,0,0});
        timer.reset();
        // viewport.render(spheres);
        viewport.render(world);
        print("Frame: {} - {}\n",frame,ms_to_human(timer.duration()));
        viewport.threaded_write_to_png(std::format("video/{}.png",frame));
    // }

    print("\n\nTotal Time {}\n",ms_to_human(totalTimer.duration()));
    return 0;
}