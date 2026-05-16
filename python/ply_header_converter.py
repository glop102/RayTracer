#!/usr/bin/env python

"""
This is a converter helper script to create a header file from a .ply file.
The header file is to be directly consumed in the c++ project instead of having it mess around with deserializing the string content.
Yes, that means the ray tracer is not being made a generic system for arbitrary scenes from a config file, but the repo is currently a toy project.
"""

import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <ply file path>")
    print("it will simply output the header file with the name 'model.hpp'")
    exit()

f = open(sys.argv[1],"r")
if not f:
    print("Cannot open file?")
    exit()

# Parse the header
number_of_verticies = 0
number_of_triangles = 0
for l in f:
    if l.strip() == "end_header":
        break
    if l.startswith("element vertex"):
        number_of_verticies = int(l.strip().split()[-1])
    if l.startswith("element face"):
        number_of_triangles = int(l.strip().split()[-1])

if not number_of_verticies:
    print("Did not parse header to find a vertex count - cannot continue")
    exit(1)
if not number_of_verticies:
    print("Did not parse header to find a triangle count - cannot continue")
    exit(1)

# Parse out the verticies
vertexes = []
for _ in range(number_of_verticies):
    l = next(f)
    l = l.split()
    vertexes.append((
        float(l[0]),
        float(l[1]),
        float(l[2]),
    ))

# Parse out the triangles
triangles = []
for _ in range(number_of_triangles):
    l = next(f)
    l = l.split()
    triangles.append((
        vertexes[int(l[1])],
        vertexes[int(l[2])],
        vertexes[int(l[3])],
    ))

out = open("model.hpp","w")
out.write("std::vector<std::shared_ptr<Triangle>> model_triangles(std::shared_ptr<Material> material){\n")
out.write("    std::vector<std::shared_ptr<Triangle>> triangles;")
for t in triangles:
    out.write(
        """
        triangles.push_back( std::make_shared<Triangle>(
            Point3 {{ {} }},
            Point3 {{ {} }},
            Point3 {{ {} }},
            material
        ));""".format(
            f"{t[0][0]},{t[0][2]},{t[0][2]}",
            f"{t[1][0]},{t[1][2]},{t[1][2]}",
            f"{t[2][0]},{t[2][2]},{t[2][2]}",
        )
    )
out.write("\n    return triangles;\n")
out.write("}")