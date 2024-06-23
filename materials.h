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