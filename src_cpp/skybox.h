#pragma once
#include <memory>
#include <string>
#include "vec_utils.h"
#include "image.h"

class Skybox {
public:
    virtual Color color(const Ray& ray) const = 0;
    virtual ~Skybox() = default;
};

class VoidSkybox : public Skybox {
public:
    Color color(const Ray& ray) const override;
};

class SkylineSkybox : public Skybox {
public:
    Color color(const Ray& ray) const override;
};

// Expects a horizontal-cross layout PNG (4 faces wide, 3 faces tall):
//       [+Y]
// [-X]  [+Z]  [+X]  [-Z]
//       [-Y]
class CubemapSkybox : public Skybox {
    Image _image;
    int _face_size;
public:
    CubemapSkybox(const std::string& path);
    Color color(const Ray& ray) const override;
private:
    Color sample(int face_col, int face_row, double s, double t) const;
};
