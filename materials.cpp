#include "materials.h"

bool Material::scatter(const Ray& incident, const HitRecord& rec, Color& attenuation, Ray& outgoing_bounce)const{
 return false;
}


// BRDMaterial::BRDMaterial(const BRDMaterial& other)
// : diffuse(other.diffuse), specular(other.specular), emissive(other.emissive), specular_tightness(other.specular_tightness), roughness(other.roughness)
// {}
BRDMaterial::BRDMaterial(const Color& diffuse, const Color& specular, const Color& emissive, const double& specular_tightness, const double& roughness)
: diffuse(diffuse), specular(specular), emissive(emissive), specular_tightness(specular_tightness), roughness(roughness)
{}

bool BRDMaterial::scatter(const Ray& incident, const HitRecord& rec, Color& attenuation, Ray& outgoing_bounce)const{
    Vector3 diffuse_cast = rec.normal + Vector3::random_unit_vector(); // lambertian diffusion
    Vector3 specular_cast = Vector3::reflect_around_normal(rec.normal,incident.direction);
    outgoing_bounce.direction = Vector3::lerp(specular_cast,diffuse_cast,roughness).normalize();
    if(outgoing_bounce.direction.near_zero()){
        outgoing_bounce.direction = specular_cast;
    }

    outgoing_bounce.origin = rec.intersection_point;

    attenuation = Vector3::lerp(specular,diffuse, outgoing_bounce.direction.dot(rec.normal) );
    return true;
}

BRDMaterial BRDMaterial::random(){
    // return {
    //     .diffuse{random_percentage_distribution(gen),random_percentage_distribution(gen),random_percentage_distribution(gen)},
    //     .specular{random_percentage_distribution(gen),random_percentage_distribution(gen),random_percentage_distribution(gen)},
    //     .emissive{random_percentage_distribution(gen),random_percentage_distribution(gen),random_percentage_distribution(gen)},
    //     .specular_tightness{random_percentage_distribution(gen)},
    //     .roughness{random_percentage_distribution(gen)}
    // };
    return BRDMaterial(
        {random_percentage_distribution(gen),random_percentage_distribution(gen),random_percentage_distribution(gen)},
        {random_percentage_distribution(gen),random_percentage_distribution(gen),random_percentage_distribution(gen)},
        {0,0,0},
        random_percentage_distribution(gen),
        random_percentage_distribution(gen)
    );
}

std::shared_ptr<BRDMaterial> AluminiumDull =
    std::make_shared<BRDMaterial>(
    Color{0.75,0.75,0.75}, //diffuse
    Color{0.9,0.9,0.9}, //specular
    Color{0,0,0}, //emissive
    0.0, //specular_tightness
    0.15 //roughness
);
std::shared_ptr<BRDMaterial> Mirror = 
    std::make_shared<BRDMaterial>(
    Color{1.0,1.0,1.0}, //diffuse
    Color{1.0,1.0,1.0}, //specular
    Color{0,0,0}, //emissive
    0.0, //specular_tightness
    0.0 //roughness
);
std::shared_ptr<BRDMaterial> MetalShiny = 
    std::make_shared<BRDMaterial>(
    Color{0.85,0.85,0.85}, //diffuse
    Color{0.9,0.9,0.9}, //specular
    Color{0,0,0}, //emissive
    0.0, //specular_tightness
    0.05 //roughness
);


//===================================================================
// PureTransparentMaterial
//===================================================================
PureTransparentMaterial::PureTransparentMaterial(const double& refractive_index)
:refractive_index(refractive_index)
{}
bool PureTransparentMaterial::scatter(const Ray& incident, const HitRecord& rec, Color& attenuation, Ray& outgoing_bounce)const{
    attenuation = White;
    double ri_ratio = rec.front_face ? (1.0/refractive_index) : refractive_index;
    double cos_theta = fmin(incident.direction.reverse().dot(rec.normal), 1.0);
    double sin_theta = sqrt(1.0 - (cos_theta*cos_theta));

    bool can_refract = (ri_ratio * sin_theta) <= 1.0;
    if(can_refract)
        outgoing_bounce.direction = rec.normal.refract(incident.direction,ri_ratio);
    else
        outgoing_bounce.direction = rec.normal.reflect(incident.direction);
    outgoing_bounce.origin = rec.intersection_point;
    return true;
}