#include "model.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

extern bool load_ply_file(std::string filename, HittableList& list, std::shared_ptr<Material> material, double scale, Point3 center) {
    std::ifstream file(filename);
    if(!file.is_open()) return false;
    unsigned int num_points=0, num_faces=0;
    std::string line;

    // Parse the header for the counts of points and triangles
    std::getline(file,line);
    while(! line.starts_with("end_header")) {
        if (line.starts_with("element vertex ")) {
            line = line.substr(strlen("element vertex "));
            num_points = std::stod(line);
        }
        if (line.starts_with("element face ")) {
            line = line.substr(strlen("element face "));
            num_faces = std::stod(line);
        }
        std::getline(file,line);
    }

    printf("%s: %d points, %d faces\n", filename.c_str(), num_points, num_faces );
    if (!num_points || !num_faces){
        printf("Not able to parse header correctly\n");
        return false;
    }

    std::vector<Point3> vertexes;
    vertexes.reserve(num_points);
    while(num_points){
        num_points--;
        double x,y,z;
        file >> x;
        file >> y;
        file >> z;
        vertexes.push_back( (Point3{x,y,z} * scale) + center );
        std::getline(file,line); // consume the rest of the line
    }

    list.objects.reserve(list.objects.size() + num_faces);
    while(num_faces){
        num_faces--;
        int num_points_in_face;
        file >> num_points_in_face;
        if (num_points_in_face != 3) {
            printf("Warning: %d points in a face importing model - currently only triangles are supported\n", num_points_in_face);
        }
        unsigned int p1,p2,p3;
        file >> p1;
        file >> p2;
        file >> p3;
        list.add(
            std::make_shared<Triangle>(
                vertexes[p1],
                vertexes[p2],
                vertexes[p3],
                material
            )
        );
        std::getline(file,line); // consume the rest of the line
    }

    return true;
}