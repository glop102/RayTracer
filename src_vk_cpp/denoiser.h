#pragma once
#include <OpenImageDenoise/oidn.hpp>
#include <cstdint>

struct Denoiser {
    Denoiser();
    void setup(uint32_t w, uint32_t h,
               void* color_ptr, void* albedo_ptr, void* normal_ptr, void* output_ptr);
    void execute();
    bool enabled = true;
private:
    oidn::DeviceRef device_;
    oidn::FilterRef filter_;
};
