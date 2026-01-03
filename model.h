#include "scene.h"

extern bool load_ply_file(std::string filename, HittableList& list, std::shared_ptr<Material> material, double scale, Point3 center);