# Toy Raytracing Project

Playing around with visual topics like raytracing is just generally pretty fun. This is my (periodically touched) implementation of cpu based ray tracing.

It has some hard-coded object setup in the main.cpp file to generate the scene to be rendered, and you will need to make a folder called "video" in the working directory. It will generate and save each frame into that video folder that you can later use ffmpeg to combine together into a video.

## examples

[Simple spin around some spheres with 500 spheres randomly generated around the orgin. (1920x1080 at 1000 rays per pixel)](https://github.com/glop102/RayTracer/raw/refs/heads/master/example_outputs/camera_sweep_2.mp4)


## building and running

The make file is a pretty simple and mostly sane build that just requires a modern gcc installation and libpng installed (with the headers which might need to be installed with the libpng-dev package on ubuntu).

It will generate a `raytrace` exectuable that you just directly run and it will print out how long each frame took to process.