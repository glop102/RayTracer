#include "denoiser.h"
#include <cstdio>

Denoiser::Denoiser() {
    device_ = oidn::newDevice(oidn::DeviceType::CPU);
    device_.commit();
}

void Denoiser::setup(uint32_t w, uint32_t h,
                     void* color_ptr, void* albedo_ptr, void* normal_ptr, void* output_ptr) {
    filter_ = device_.newFilter("RT");
    // pixelByteStride=16: RGBA32F — read only RGB, skip A.
    filter_.setImage("color",  color_ptr,  oidn::Format::Float3, w, h, 0, 16);
    filter_.setImage("albedo", albedo_ptr, oidn::Format::Float3, w, h, 0, 16);
    filter_.setImage("normal", normal_ptr, oidn::Format::Float3, w, h, 0, 16);
    filter_.setImage("output", output_ptr, oidn::Format::Float3, w, h, 0, 16);
    filter_.set("hdr", true);
    filter_.set("quality", oidn::Quality::Balanced);
    filter_.commit();
}

void Denoiser::execute() {
    filter_.execute();
    const char* err;
    if (device_.getError(err) != oidn::Error::None)
        fprintf(stderr, "OIDN: %s\n", err);
}
