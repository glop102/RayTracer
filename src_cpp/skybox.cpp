#include "skybox.h"
#include <cmath>
#include <algorithm>

Color VoidSkybox::color(const Ray&) const {
    return Black;
}

Color SkylineSkybox::color(const Ray& ray) const {
    auto y = ray.direction.y / ray.direction.length();
    if (y > 0)
        return Vector3::lerp(White, BlueSky, y);
    else if (y > -0.5)
        return White * (1.0 + (y * 2));
    else
        return Black;
}

CubemapSkybox::CubemapSkybox(const std::string& path)
    : _image(Image::read_from_png(path)), _face_size(_image.height() / 3)
{}

// s, t in [-1, 1] -> pixel in the given face cell of the cross layout
Color CubemapSkybox::sample(int face_col, int face_row, double s, double t) const {
    int u = (int)std::round((s + 1.0) * 0.5 * (_face_size - 1));
    int v = (int)std::round((t + 1.0) * 0.5 * (_face_size - 1));
    u = std::clamp(u, 0, _face_size - 1);
    v = std::clamp(v, 0, _face_size - 1);
    int px = face_col * _face_size + u;
    int py = face_row * _face_size + v;
    return _image[py * _image.width() + px];
}

Color CubemapSkybox::color(const Ray& ray) const {
    const Vector3& d = ray.direction;
    double ax = std::abs(d.x), ay = std::abs(d.y), az = std::abs(d.z);

    int face_col, face_row;
    double s, t;

    if (ax >= ay && ax >= az) {
        if (d.x > 0) { face_col = 2; face_row = 1; s = -d.z/ax; t = -d.y/ax; } // +X
        else          { face_col = 0; face_row = 1; s =  d.z/ax; t = -d.y/ax; } // -X
    } else if (ay >= ax && ay >= az) {
        if (d.y > 0) { face_col = 1; face_row = 0; s =  d.x/ay; t =  d.z/ay; } // +Y
        else          { face_col = 1; face_row = 2; s =  d.x/ay; t = -d.z/ay; } // -Y
    } else {
        if (d.z > 0) { face_col = 1; face_row = 1; s =  d.x/az; t = -d.y/az; } // +Z
        else          { face_col = 3; face_row = 1; s = -d.x/az; t = -d.y/az; } // -Z
    }

    return sample(face_col, face_row, s, t);
}
