#pragma once
#include "vec_utils.h"
#include <vector>
#include <string>

using Color = Vector3;

class Image:public std::vector<Color>{
    protected:
    int _height,_width;

    public:
    Image(int width, int height);
    int width();
    int height();

    Color& get_px(const int& x,const int& y);

    void write_to_png(std::string filename)const;
    static double linear_to_gamma(double px);
};

extern const Color White,Red,Green,Blue,Grey,Black;
extern const Color DarkRed,DarkGreen,DarkBlue;
extern const Color BlueSky;