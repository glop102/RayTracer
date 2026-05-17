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
    int num_frames = 120;
};

static void print_usage(const char* prog) {
    print("Usage: {} [options]\n", prog);
    print("  --width  N       Image width  (default 1920)\n");
    print("  --height N       Image height (default 1080)\n");
    print("  --rays   N       Rays per pixel (default 100)\n");
    print("  --depth  N       Max ray bounce depth (default 10)\n");
    print("  --frames N       Number of frames to render (default 120)\n");
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
        else if (std::strcmp(argv[i], "--frames") == 0) cfg.num_frames = need_next();
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

void populate_cornell_box(HittableList& list, const BBox& room) {
    auto white = std::make_shared<BRDMaterial>(Color{0.73,0.73,0.73}, Color{0.73,0.73,0.73}, Black, 0.0, 1.0);
    auto red   = std::make_shared<BRDMaterial>(Color{0.65,0.05,0.05}, Color{0.65,0.05,0.05}, Black, 0.0, 1.0);
    auto green = std::make_shared<BRDMaterial>(Color{0.12,0.45,0.15}, Color{0.12,0.45,0.15}, Black, 0.0, 1.0);
    auto light = std::make_shared<BRDMaterial>(Black, Black, Color{12,12,12}, 0.0, 1.0);
    auto glass = std::make_shared<PureTransparentMaterial>(1.5);

    const Point3& lo = room.min;
    const Point3& hi = room.max;

    // Normals for all walls point INTO the room.
    // Room opens toward camera at -Z; back wall is at hi.z.
    // Winding verified by e1×e2 cross product for each face.

    // Back wall (z=hi.z), normal -Z: top-left → top-right → bot-right → bot-left
    for (auto& t : make_quad({lo.x,hi.y,hi.z},{hi.x,hi.y,hi.z},{hi.x,lo.y,hi.z},{lo.x,lo.y,hi.z}, white)) list.add(t);
    // Floor (y=lo.y), normal +Y: front-left → front-right → back-right → back-left
    for (auto& t : make_quad({lo.x,lo.y,lo.z},{hi.x,lo.y,lo.z},{hi.x,lo.y,hi.z},{lo.x,lo.y,hi.z}, white)) list.add(t);
    // Ceiling (y=hi.y), normal -Y: front-left → back-left → back-right → front-right
    for (auto& t : make_quad({lo.x,hi.y,lo.z},{lo.x,hi.y,hi.z},{hi.x,hi.y,hi.z},{hi.x,hi.y,lo.z}, white)) list.add(t);
    // Left wall (x=lo.x, red), normal +X: back-bot → front-bot → front-top → back-top
    for (auto& t : make_quad({lo.x,lo.y,hi.z},{lo.x,lo.y,lo.z},{lo.x,hi.y,lo.z},{lo.x,hi.y,hi.z}, red)) list.add(t);
    // Right wall (x=hi.x, green), normal -X: front-bot → back-bot → back-top → front-top
    for (auto& t : make_quad({hi.x,lo.y,lo.z},{hi.x,lo.y,hi.z},{hi.x,hi.y,hi.z},{hi.x,hi.y,lo.z}, green)) list.add(t);

    // Ceiling light: small panel slightly inset, normal -Y (same winding as ceiling)
    double lx1 = lo.x + (hi.x-lo.x)*0.3,  lx2 = lo.x + (hi.x-lo.x)*0.7;
    double lz1 = lo.z + (hi.z-lo.z)*0.3,  lz2 = lo.z + (hi.z-lo.z)*0.7;
    double ly  = hi.y - 0.01;
    for (auto& t : make_quad({lx1,ly,lz1},{lx1,ly,lz2},{lx2,ly,lz2},{lx2,ly,lz1}, light)) list.add(t);

    // Load bunny in object space (no translation offset — Instance handles placement)
    HittableList bunny;
    double scale = (hi.y - lo.y) * 0.55;
    load_ply_file("bunny/reconstruction/bun_zipper.ply", bunny, AluminiumDull, scale, {0,0,0});

    // Glass prism snugly around the bunny in object space
    BBox bunny_bbox = bunny.bbox();
    double pad = (hi.y - lo.y) * 0.02;
    BBox padded{
        {bunny_bbox.min.x-pad, bunny_bbox.min.y-pad, bunny_bbox.min.z-pad},
        {bunny_bbox.max.x+pad, bunny_bbox.max.y+pad, bunny_bbox.max.z+pad}
    };
    for (auto& t : make_box(padded, glass)) bunny.add(t);

    // Build a single Instance from the combined bunny + glass geometry,
    // then place it: sitting on the floor, centered in x and z.
    auto inst = std::make_shared<Instance>(bunny.objects);
    BBox obj_bbox = inst->bbox(); // identity transform, so same as raw bbox
    inst->translate({
        (lo.x+hi.x)/2.0 - obj_bbox.center().x,
        lo.y            - obj_bbox.min.y,
        (lo.z+hi.z)/2.0 - obj_bbox.center().z,
    });
    list.add(inst);
}

void populate_triangles_crafted_test(HittableList& list){
    auto glass = std::make_shared<PureTransparentMaterial>(1.5);

    // Load bunny into a temporary list so we can measure its bbox
    HittableList bunny_list;
    load_ply_file("bunny/reconstruction/bun_zipper.ply", bunny_list, AluminiumDull, 100.0, Point3{0,-5.0,0});

    // Build a glass box snugly around the bunny using its actual bbox
    BBox bunny_bbox = bunny_list.bbox();
    // Add a small padding so the bunny doesn't clip the glass surface
    double pad = 0.3;
    BBox padded{
        {bunny_bbox.min.x - pad, bunny_bbox.min.y - pad, bunny_bbox.min.z - pad},
        {bunny_bbox.max.x + pad, bunny_bbox.max.y + pad, bunny_bbox.max.z + pad}
    };
    for (auto& tri : make_box(padded, glass))
        list.add(tri);

    // Add bunny triangles to main list
    for (auto& obj : bunny_list.objects)
        list.add(obj);
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
    print("Config: {}x{}, {} rays/pixel, depth {}, {} frames\n", cfg.width, cfg.height, cfg.rays_per_pixel, cfg.ray_depth, cfg.num_frames);

    // Camera viewport(1920*4,1080*4);
    Camera viewport(cfg.width, cfg.height);
    // Camera viewport(1920/2,1080/2);
    viewport.sampling_per_pixel = cfg.rays_per_pixel;
    viewport.max_trace_depth    = cfg.ray_depth;
    // viewport.sampling_per_pixel = 1000;
    // viewport.ongoing_image_export = 32;
    HittableList spheres;
    BBox cornell_room{{-5, 0, 0}, {5, 10, 10}};
    populate_cornell_box(spheres, cornell_room);

    Stopwatch timer,totalTimer;
    BVH8List world(spheres.objects);
    print("BVH Creation Time  {}\n",timer.duration());
    world.debug_print_tree();

    // Camera orbits around the Cornell box looking at its center.
    // Room opens toward -Z, so orbit stays at z <= -5 to look in from the front.
    Vector3 room_center{0, 5, 5};
    for(int frame=0; frame < cfg.num_frames; frame++){
        double t = frame / (double)cfg.num_frames;
        // Gentle arc: swing ±40° horizontally in front of the opening
        double angle = (t - 0.5) * 2.0 * 0.7; // -0.7..+0.7 radians
        double dist = 18.0;
        viewport.origin = Vector3{std::sin(angle)*dist, 5, -std::cos(angle)*dist + room_center.z};
        viewport.look_at(room_center);
        timer.reset();
        viewport.render(world);
        print("Frame: {} - {}\n",frame,ms_to_human(timer.duration()));
        viewport.threaded_write_to_png(std::format("video/{}.png",frame));
    }

    print("\n\nTotal Time {}\n",ms_to_human(totalTimer.duration()));
    return 0;
}
