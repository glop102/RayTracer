#pragma once
#include <string>
#include <thread>
#include "image.h"
#include "vec_utils.h"
#include "utils.h"
#include "scene.h"


class Camera{
    private:
    std::vector<std::jthread> image_save_threads;
    public:
    Image* pixels;
    Vector3 origin = {0.0,0.0,0.0};
    Vector3 look_direction = {0.0,0.0,-1.0};
    Vector3 up_direction = {0.0,1.0,0.0};
    double focal_length, viewport_height; // the distanceScale from the center and the 'sensor size' of the screen - relates to FOV
    bool inverted_y = true; // the world has up as positive y, but most viewports have positive y going down
    int sampling_per_pixel = 100;
    int max_trace_depth = 10;
    int ongoing_image_export = 0;

    protected:
    Vector3 viewport_up,viewport_right; // Calculated at the start of the render based on the look and up directions
    Ray _initial_pixel_ray(int x, int y, Vector3& screen_origin, Vector3& pixel_delta_x, Vector3& pixel_delta_y, double x_variance=0.0, double y_variance=0.0)const;
    Vector3 _calculate_screen_origin();
    Vector3 _calculate_pixel_delta_x()const;
    Vector3 _calculate_pixel_delta_y()const;

    void _scanline_thread_runner(const Hittable& scene, int y);
    Color _cast_ray_for_color(const Ray& ray, const Hittable& scene, int max_depth, Color total_attenuation);

    public:
    Camera(int px_width=1920, int px_height=1080, double focal_length=1.0, double viewport_height=2.0);
    ~Camera();

    void look_at(const Vector3& point);
    void write_to_png(std::string)const;
    void threaded_write_to_png(std::string);
    void debug_print();

    double viewport_width()const;

    void render(const Hittable& scene);
};