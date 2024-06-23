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
    this->up_direction = look_direction.cross(up_direction).cross(look_direction).normalize();
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
    up_direction.normalize();
    look_direction.normalize();
    double viewport_width = (pixels->width()/(double)pixels->height()) * viewport_height;
    Vector3 left_direction = up_direction.cross(look_direction).normalize();
    Vector3 center_of_sensor = origin + (look_direction*focal_length);
    if(inverted_y)
        return center_of_sensor + (left_direction*(viewport_width)/2.0) + (up_direction*(viewport_height/2.0));
    return center_of_sensor + (left_direction*(viewport_width)/2.0) + (up_direction*(viewport_height/-2.0));
}
Vector3 Camera::_calculate_pixel_delta_x()const{
    Vector3 right_direction = look_direction.cross(up_direction).normalize(); // points to the right of the sensor
    return  right_direction* (viewport_width()/(pixels->width()-1));
}
Vector3 Camera::_calculate_pixel_delta_y()const{
    if(inverted_y)
        return up_direction*(-1 * viewport_height/(pixels->height()-1)) ;
    return up_direction*(viewport_height/(pixels->height()-1)) ;
}

double Camera::viewport_width()const{
    return (pixels->width()/(double)pixels->height()) * viewport_height; // aspect ratio times the virtual sensor height
}

void Camera::render(const Hittable& scene){
    // Spawns multiple threads to saturate a CPU
    // Each thread locks the atomic to safely pick what is the next line that will be rendered
    // The real work of the ray tracing is done in the function call in the lambda
    // Note: there is no lock on the pixels array because each thread should be independantly writing to their pixels and should not step on each other
    std::atomic<int> next_line = 0;
    auto capture = [this,&scene,&next_line](){
            int max_line = pixels->height();
            int our_claimed_line = next_line++;
            while(our_claimed_line < max_line){
                this->_scanline_thread_runner(scene,our_claimed_line);
                if(ongoing_image_export && our_claimed_line %ongoing_image_export == 0)
                    write_to_png("ongoing.png");
                our_claimed_line = next_line++;
            }
        };
    int max_threads = thread::hardware_concurrency();
    std::vector<std::jthread> threads(max_threads);
    for(int t=0; t<max_threads; t++){
        threads.emplace_back( std::jthread(capture) );
    }
}

void Camera::_scanline_thread_runner(const Hittable& scene, int y){
    // Renders a full line its whole width
    auto screen_origin = _calculate_screen_origin();
    auto pixel_delta_x = _calculate_pixel_delta_x();
    auto pixel_delta_y = _calculate_pixel_delta_y();
    for(int x=0;x<pixels->width(); x++){
        Color accum = Black;
        for(int sample=0; sample<sampling_per_pixel; sample++){
            Ray ray = _initial_pixel_ray(x,y,screen_origin,pixel_delta_x,pixel_delta_y, random_neg_pos_one(gen)/2.0, random_neg_pos_one(gen)/2.0);
            accum += _cast_ray_for_color(ray,scene, max_trace_depth);
        }
        pixels->get_px(x,y) = accum / sampling_per_pixel;
    }
}

Color Camera::_cast_ray_for_color(const Ray& ray, const Hittable& scene, int max_depth){
    HitRecord rec;
    RealRange initial_allowed_range(std::numeric_limits<double>::epsilon()*10.0,Infinity);
    if(max_depth<=0){
        return Black;
    }else if(scene.hit(ray,initial_allowed_range,rec)){
        Color attenuation;
        Ray next_bounce;
        if(! rec.material->scatter(ray,rec,attenuation,next_bounce)){
            return Black;
        }
        Color incoming_color = _cast_ray_for_color(next_bounce, scene, max_depth-1 );
        
        return attenuation * incoming_color;
    }else{
        // Lets simulate a light blue skybox gradient if we completly miss.
        // It is ever so slightly faster to normalize just our Y component since that is all we need
        auto y = ray.direction.y/ray.direction.length();

        // a skybox that is actually blue up top, white at the horizon, and void underneith
        if(y>0){
            return Vector3::lerp( White, BlueSky, y);
        }else{
            return White*(1.0+y);
        }
    }
}