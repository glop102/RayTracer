#include "image.h"
#define PNG_SETJMP_NOT_SUPPORTED
#include <png.h>
#include <zlib.h>
#include <cstdlib>

const Color White={1.0,1.0,1.0};
const Color Red=  {1.0,0.0,0.0};
const Color Green={0.0,1.0,0.0};
const Color Blue= {0.0,0.0,1.0};
const Color Grey= {0.5,0.5,0.5};
const Color Black={0.0,0.0,0.0};
const Color DarkRed=  {0.5,0.0,0.0};
const Color DarkGreen={0.0,0.5,0.0};
const Color DarkBlue= {0.0,0.0,0.5};
const Color BlueSky = {0.4,0.6,0.9};

Image::Image(int width, int height): std::vector<Color>(height*width,{0.0,0.0,0.0}), _height(height), _width(width){}
int Image::width() const {return _width;}
int Image::height() const {return _height;}

Color& Image::get_px(const int& x,const int& y){
    return this->operator[]((y*_width) + x);
}

png_byte png_clamp(int val){
    if(val<0) return 0;
    if(val>255) return 255;
    return val;
}

void Image::write_to_png(std::string filename)const{
    FILE* fp = fopen(filename.c_str(),"wb");
    if(!fp) return;

    //Initializa Libpng
    png_structp png_ptr = png_create_write_struct(
        PNG_LIBPNG_VER_STRING,
        nullptr, //(png_voidp)user_error_ptr,
        nullptr, //user_error_fn,
        nullptr //user_warning_fn
        );
    if (!png_ptr)
        return;
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr){
        png_destroy_write_struct(&png_ptr,
            (png_infopp)nullptr);
        fclose(fp);
        return;
    }

    //Lets setup for writing
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr,
        _width,
        _height,
        8, //bitdepth
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);
    /* set the zlib compression level */
    // png_set_compression_level(png_ptr, Z_BEST_COMPRESSION);
    png_set_compression_level(png_ptr, 2); // to make it save images faster, tell it to do less compression
    /* set other zlib parameters */
    png_set_compression_mem_level(png_ptr, 8);
    png_set_compression_strategy(png_ptr, Z_DEFAULT_STRATEGY);
    png_set_compression_window_bits(png_ptr, 15);
    png_set_compression_method(png_ptr, 8);
    png_set_compression_buffer_size(png_ptr, 8192);
    // png_set_filter(png_ptr,0,PNG_FILTER_SUB); // restricting the filtering methods available doesn't improve processing speed much

    //Start writing
    png_write_info(png_ptr, info_ptr);
    std::vector<png_byte> rowbuf(_width*3); // *3 for RGB
    for(int row=0; row<_height; row++){
        //copy into the temp buf
        for(int x=0; x<_width; x++){
            Color px = operator[](row*_width + x);
            rowbuf[x*3 + 0] = png_clamp( linear_to_gamma(px.red) * 255 );
            rowbuf[x*3 + 1] = png_clamp( linear_to_gamma(px.green) * 255 );
            rowbuf[x*3 + 2] = png_clamp( linear_to_gamma(px.blue) * 255 );
        }
        png_write_row(png_ptr,rowbuf.data());
    }

    //finalize writing
    png_write_end(png_ptr, info_ptr);

    //cleanup
    png_destroy_write_struct(&png_ptr,&info_ptr);
    fclose(fp);
}

double Image::linear_to_gamma(double px){
    return sqrt(px);
}

Image Image::read_from_png(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", filename.c_str());
        std::exit(1);
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) { fclose(fp); std::exit(1); }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { png_destroy_read_struct(&png_ptr, nullptr, nullptr); fclose(fp); std::exit(1); }

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    int width  = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth  = png_get_bit_depth(png_ptr, info_ptr);

    if (bit_depth == 16)                                          png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE)                     png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)       png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))          png_set_tRNS_to_alpha(png_ptr);
    if (color_type & PNG_COLOR_MASK_ALPHA)                        png_set_strip_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)                  png_set_gray_to_rgb(png_ptr);

    png_read_update_info(png_ptr, info_ptr);

    Image img(width, height);
    std::vector<png_byte> rowbuf(width * 3);
    for (int y = 0; y < height; y++) {
        png_read_row(png_ptr, rowbuf.data(), nullptr);
        for (int x = 0; x < width; x++) {
            // sRGB -> linear: undo the sqrt gamma applied on write
            double r = rowbuf[x*3 + 0] / 255.0;
            double g = rowbuf[x*3 + 1] / 255.0;
            double b = rowbuf[x*3 + 2] / 255.0;
            img.get_px(x, y) = {r*r, g*g, b*b};
        }
    }

    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(fp);
    return img;
}