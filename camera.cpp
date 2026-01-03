#include "camera.h"
#include <thread>
#include <random>
#include <algorithm>
#include <execution>
#include <ranges>
#include <atomic>

using std::thread;

Camera::Camera(int px_width, int px_height, double focal_length, double viewport_height):
    pixels(new Image(px_width,px_height)), focal_length(focal_length), viewport_height(viewport_height)
{}

Camera::~Camera(){
    delete pixels;
    image_save_threads.clear(); // will join each thread as it gets cleared
}

void Camera::look_at(const Vector3 &point){
    this->look_direction = (point-origin).normalize();
    // this->up_direction = look_direction.cross(up_direction).cross(look_direction).normalize();
}

void Camera::write_to_png(std::string filename)const{
    pixels->write_to_png(filename);
}
void Camera::threaded_write_to_png(std::string filename){
    int px_width = pixels->width();
    int px_height = pixels->height();
    image_save_threads.emplace_back(
        std::jthread([](Image* img,std::string filename){
            img->write_to_png(filename);
            delete img;
        },
        pixels,filename
    ));
    pixels = new Image(px_width,px_height);
}

void Camera::debug_print(){
    print("Camera: {:.3f} {:.3f} {:.3f} -> {:.3f} {:.3f} {:.3f}\n",origin.x,origin.y,origin.z,look_direction.x,look_direction.y,look_direction.z);
    auto o = _calculate_screen_origin();
    print("Screen: {:.3f} {:.3f} {:.3f}   Size: w{:.2f} h{:.2f}\n",o.x,o.y,o.z,viewport_width(),viewport_height);
    auto dx = _calculate_pixel_delta_x();
    auto dy = _calculate_pixel_delta_y();
    print("    dx: {:.5f} {:.5f} {:.5f}   dy: {:.5f} {:.5f} {:.5f}\n",dx.x,dx.y,dx.z,dy.x,dy.y,dy.z);
}

Ray Camera::_initial_pixel_ray(int x, int y, Vector3& screen_origin, Vector3& pixel_delta_x, Vector3& pixel_delta_y, double x_variance, double y_variance)const{
    Vector3 pixel_center = screen_origin + (pixel_delta_x * x) + (pixel_delta_y * y);
    pixel_center += pixel_delta_x*x_variance + pixel_delta_y*y_variance;
    return {
        origin,
        (pixel_center - origin).normalize()
        };
}

Vector3 Camera::_calculate_screen_origin(){
    Vector3 center_of_sensor = origin + (look_direction*focal_length);
    if(inverted_y)
        return center_of_sensor + (viewport_right.reverse()*(viewport_width())/2.0) + (viewport_up*(viewport_height/2.0));
    return center_of_sensor + (viewport_right.reverse()*(viewport_width())/2.0) + (viewport_up*(viewport_height/-2.0));
}
Vector3 Camera::_calculate_pixel_delta_x()const{
    return  viewport_right* (viewport_width()/(pixels->width()-1));
}
Vector3 Camera::_calculate_pixel_delta_y()const{
    if(inverted_y)
        return viewport_up*(-1 * viewport_height/(pixels->height()-1));
    return viewport_up*(viewport_height/(pixels->height()-1));
}

double Camera::viewport_width()const{
    return (pixels->width()/(double)pixels->height()) * viewport_height; // aspect ratio times the virtual sensor height
}

void Camera::render(const Hittable& scene){
    viewport_right = look_direction.cross(up_direction).normalize();
    viewport_up = viewport_right.cross(look_direction).normalize();
    look_direction = look_direction.normalize();

    auto screen_origin = _calculate_screen_origin();
    auto pixel_delta_x = _calculate_pixel_delta_x();
    auto pixel_delta_y = _calculate_pixel_delta_y();

    // Spawns multiple threads to saturate a CPU
    // Each thread will grab the mutex to figure out what pixel they are working on
    // there is no lock on the pixel array since each thread works on a different pixel and should not step on each other
    std::atomic<int> next_x = 0;
    std::atomic<int> next_y = 0;
    std::mutex pixel_progress_lock;
    auto capture = [this, &scene, &next_y,&next_x, &pixel_progress_lock, &screen_origin,&pixel_delta_x,&pixel_delta_y](){
            int our_claimed_y, our_claimed_x;
            Color accum = Black;
            goto init_pixel_loop;
            do {
                accum = Black;
                for(int sample=0; sample<sampling_per_pixel; sample++){
                    Ray ray = _initial_pixel_ray(our_claimed_x,our_claimed_y,screen_origin,pixel_delta_x,pixel_delta_y, random_neg_pos_one(gen)/2.0, random_neg_pos_one(gen)/2.0);
                    accum += _cast_ray_for_color(ray,scene);
                }
                pixels->get_px(our_claimed_x,our_claimed_y) = accum / sampling_per_pixel;

                if(ongoing_image_export && !our_claimed_x && our_claimed_y %ongoing_image_export == 0)
                    write_to_png("ongoing.png");

                // To be actually C complient, we cannot have an infinite while loop
                // Also it just makes sense to have the loop conditional be in the right place
                // So we have this label to jump to the bookkeeping code for the loop at the end which also serves to initialize the loop conditionals
                // otherwise, i would be forced to duplicate code that does the setup before the loop
                init_pixel_loop:
                pixel_progress_lock.lock();
                our_claimed_x = next_x++;
                our_claimed_y = next_y;
                if ( our_claimed_x >= pixels->width() ) {
                    next_x = our_claimed_x = 0;
                    our_claimed_y = next_y++;
                }
                pixel_progress_lock.unlock();
            } while( our_claimed_y < this->pixels->height() );
        };
    int max_threads = thread::hardware_concurrency();
    std::vector<std::jthread> threads(max_threads);
    for(int t=0; t<max_threads; t++){
        threads.emplace_back( std::jthread(capture) );
    }
}

static inline Color simulated_skybox(const Ray& ray) {
    // Lets simulate a light blue skybox gradient if we completly miss.
    // It is ever so slightly faster to normalize just our Y component since that is all we need
    auto y = ray.direction.y/ray.direction.length();

    // a skybox that is actually blue up top, white at the horizon, and void underneith
    if(y>0){
        return Vector3::lerp( White, BlueSky, y);
    }else if(y>-0.5){
        return White * (1.0+(y*2));
    }else{
        return Black;
    }
}

Color Camera::_cast_ray_for_color(Ray& ray, const Hittable& scene){
    HitRecord rec;
    Color total_attenuation = White;
    Color accumulated_energy = Black;
    int depth_left = this->max_trace_depth;
    while (depth_left > 0 && total_attenuation.length_squared()>0.0000001){
        // This gets remade every loop since the .hit() method will trim the allowed_range to find only closer hits as it goes
        RealRange hit_allowed_range(0.0001,Infinity);
        if(scene.hit(ray,hit_allowed_range,rec)){
            depth_left--;
            Color additional_attenuation;
            Ray next_bounce;
            rec.material->scatter(ray,rec,additional_attenuation,next_bounce);

            accumulated_energy += total_attenuation * rec.material->extra_light(ray,rec,total_attenuation);
            total_attenuation = total_attenuation * additional_attenuation;
            ray = next_bounce;
        } else {
            accumulated_energy += total_attenuation * simulated_skybox(ray);
            break;
        }
    }
    return accumulated_energy;
}
