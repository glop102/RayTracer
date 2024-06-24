#pragma once
#include "vec_utils.h"
#include "shapes.h"
#include "image.h"

class HitRecord;

class Material{
    public:
    virtual bool scatter(const Ray& incident, const HitRecord& rec, Color& attenuation, Ray& outgoing_bounce)const;
};

class BRDMaterial:public Material{
    public:
    Color diffuse;
    Color specular;
    Color emissive;
    double specular_tightness;
    double roughness;

    // BRDMaterial(const BRDMaterial& other);
    BRDMaterial(const Color& diffuse, const Color& specular, const Color& emissive, const double& specular_tightness, const double& roughness);
    bool scatter(const Ray& incident, const HitRecord& rec, Color& attenuation, Ray& outgoing_bounce)const;
    static BRDMaterial random();
};

extern std::shared_ptr<BRDMaterial> AluminiumDull;
extern std::shared_ptr<BRDMaterial> Mirror;
extern std::shared_ptr<BRDMaterial> MetalShiny;

class PureTransparentMaterial:public Material{
    public:
    // RELATIVE!!! Refractive index. If you have a material embedded in another material, then this should be the ratio between those materials
    //1.0 is Air, and higher values for other materials, notably glass is 1.5-1.7, water is 1.33
    double refractive_index;

    PureTransparentMaterial(const double& refractive_index);
    bool scatter(const Ray& incident, const HitRecord& rec, Color& attenuation, Ray& outgoing_bounce)const;
};